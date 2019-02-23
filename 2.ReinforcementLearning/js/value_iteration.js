var RL_VI = {};

(function(global) {
  "use strict";

  // syntactic sugar function for getting default parameter values
  var getopt = function(opt, field_name, default_value) {
    if (typeof opt === 'undefined') { return default_value; }
    return (typeof opt[field_name] !== 'undefined') ? opt[field_name] : default_value;
  }

  var zeros = R.zeros; // inherit these
  var assert = R.assert;
  var randi = R.randi;
  var randf = R.randf;

  var setConst = function(arr, c) {
    for(var i=0,n=arr.length;i<n;i++) {
      arr[i] = c;
    }
  }

  // p = [0.1, 0.4, 0.4, 0.1] - Uniform Distribution 에 따르는 Random Action 추출
  var sampleWeighted = function(p) {
    var r = Math.random();
    var c = 0.0;
    for(var i=0; i<p.length; i++) {
      c += p[i];
      if(c >= r) { return i; }
    }
    assert(false, 'wtf');
  }

  // ------
  // AGENTS
  // ------

  // DPAgent performs Value Iteration
  // - can also be used for Policy Iteration if you really wanted to
  // - requires model of the environment :(
  // - does not learn from experience :(
  // - assumes finite MDP :(
  var DPAgent = function(env, opt) {
    this.V = null; // state value function
    this.P = null; // policy distribution \pi(s,a)
    this.env = env; // store pointer to environment
    this.gamma = getopt(opt, 'gamma', 0.75); // future reward discount factor
    this.reset();
  }

  DPAgent.prototype = {
    reset: function() {
      // reset the agent's policy and value function
      this.ns = this.env.getNumStates();
      this.na = this.env.getMaxNumActions();
      this.V = zeros(this.ns);
      this.P = zeros(this.ns * this.na);

      // initialize uniform random policy
      for(var s=0;s<this.ns;s++) {
        var poss = this.env.allowedActions(s);
        for(var i=0; i<poss.length; i++) {
          this.P[poss[i]*this.ns + s] = 1.0 / poss.length;
        }
      }
    },

    act: function(s) {
      // behave according to the learned policy
      var poss = this.env.allowedActions(s);
      var ps = [];
      for(var i=0; i<poss.length; i++) {
        var a = poss[i];
        var prob = this.P[a*this.ns + s];
        ps.push(prob);
      }
      var maxi = sampleWeighted(ps);
      return poss[maxi];
    },

    learn: function() {
      // perform a single round of value iteration
      self.evaluatePolicy(); // writes this.V
      self.updatePolicy(); // writes this.P
    },

    evaluatePolicy: function() {
      // perform a synchronous update of the value function
      var Vnew = zeros(this.ns);
      for(var s=0; s<this.ns; s++) {
        // integrate over actions in a stochastic policy
        // note that we assume that policy probability mass over allowed actions sums to one
        var v = 0.0;
        var poss = this.env.allowedActions(s);
        for(var i=0; i<poss.length; i++) {
          var a = poss[i];
          var prob = this.P[a*this.ns+s]; // probability of taking action under policy
          if(prob === 0) { continue; } // no contribution, skip for speed
          var ns = this.env.nextStateDistribution(s, a);
          var rs = this.env.reward(s, a, ns); // reward for s->a->ns transition
          v += prob * (rs + this.gamma * this.V[ns]);
        }
        Vnew[s] = v;
      }
      this.V = Vnew; // swap
    },

    updatePolicy: function() {
      // update policy to be greedy w.r.t. learned Value function
      for(var s=0; s<this.ns; s++) {
        var poss = this.env.allowedActions(s);

        // compute value of taking each allowed action
        var vmax, nmax;
        var vs = [];
        for(var i=0; i<poss.length; i++) {
          var a = poss[i];
          var ns = this.env.nextStateDistribution(s,a);
          var rs = this.env.reward(s, a, ns);
          var v = rs + this.gamma * this.V[ns];
          vs.push(v);
          if(i === 0 || v > vmax) { vmax = v; nmax = 1; }
          else if(v === vmax) { nmax += 1; }
        }

        // update policy smoothly across all argmaxy actions
        for(var i=0; i<poss.length; i++) {
          var a = poss[i];
          this.P[a*this.ns + s] = (vs[i] === vmax) ? 1.0/nmax : 0.0;
        }
      }
    },
  }


  // QAgent uses TD (Q-Learning, SARSA)
  // - does not require environment model :)
  // - learns from experience :)
  var TDAgent = function(env, opt) {
    this.update = getopt(opt, 'update', 'qlearn'); // qlearn | sarsa
    this.gamma = getopt(opt, 'gamma', 0.75); // future reward discount factor
    this.epsilon = getopt(opt, 'epsilon', 0.1); // for epsilon-greedy policy
    this.alpha = getopt(opt, 'alpha', 0.01); // value function learning rate

    // class allows non-deterministic policy, and smoothly regressing towards the optimal policy based on Q
    this.smooth_policy_update = getopt(opt, 'smooth_policy_update', false);
    this.beta = getopt(opt, 'beta', 0.01); // learning rate for policy, if smooth updates are on

    // eligibility traces
    this.lambda = getopt(opt, 'lambda', 0); // eligibility trace decay. 0 = no eligibility traces used
    this.replacing_traces = getopt(opt, 'replacing_traces', true);

    // optional optimistic initial values
    this.q_init_val = getopt(opt, 'q_init_val', 0);

    this.planN = getopt(opt, 'planN', 0); // number of planning steps per learning iteration (0 = no planning)

    this.Q = null; // state action value function
    this.V = null; // state value function
    this.P = null; // policy distribution \pi(s,a)
    this.e = null; // eligibility trace
    this.env_model_s = null;; // environment model (s,a) -> (s',r)
    this.env_model_r = null;; // environment model (s,a) -> (s',r)
    this.env = env; // store pointer to environment
    this.reset();
  }

  // exports
  global.DPAgent = DPAgent;
})(RL_VI);

var agent_vi, env_vi;

// gridword_vi
var gridword_vi = function(){
    this.Rarr = null; // reward array
    this.T = null; // cell types, 0 = normal, 1 = cliff
    this.reset()
};

gridword_vi.prototype = {
    reset: function() {
        // hardcoding one gridworld for now
        this.gh = 7;
        this.gw = 7;
        this.gs = this.gh * this.gw; // number of states

        // specify some rewards
        this.Rarr = R.zeros(this.gs);
        this.T = R.zeros(this.gs);

        this.Rarr[16] = -5.0;
        this.Rarr[32] = 5.0;
    },

    reward: function(s, a, ns) {
        // reward of being in s, taking action a, and ending up in ns
        return this.Rarr[ns];
    },

    nextStateDistribution: function(s,a) {
        // given (s,a) return distribution over s' (in sparse form)
        // ordinary space
        var nx, ny;
        var x = this.stox(s);
        var y = this.stoy(s);

        if (a === 0) {nx=x-1; ny=y;}
        if (a === 1) {nx=x; ny=y-1;}
        if (a === 2) {nx=x; ny=y+1;}
        if (a === 3) {nx=x+1; ny=y;}

        var ns = nx * this.gh + ny;

        // gridworld is deterministic, so return only a single next state
        return ns;
    },

    sampleNextState: function(s,a) {
        // gridworld is deterministic, so this is easy
        var ns = this.nextStateDistribution(s,a);
        var r = this.Rarr[s]; // observe the raw reward of being in s, taking a, and ending up in ns
        r -= 0.01; // every step takes a bit of negative reward
        var out = {'ns':ns, 'r':r};
        return out;
    },

    allowedActions: function(s) {
        var x = this.stox(s);
        var y = this.stoy(s);
        var as = [];
        if(x > 0) { as.push(0); } // up
        if(y > 0) { as.push(1); } // left
        if(y < this.gh-1) { as.push(2); } // right
        if(x < this.gw-1) { as.push(3); } // down
        return as;
    },

    randomState: function() {
        return Math.floor(Math.random() * this.gs);
    },

    startState: function() {
        return 0;
    },

    getNumStates: function() {
        return this.gs;
    },

    getMaxNumActions: function() {
        return 4;
    },

    // private functions
    stox: function(s) { return Math.floor(s / this.gw); },
    stoy: function(s) { return s % this.gh; },
    xytos: function(x, y) { return x * this.gw + y; },
}

// ------
// UI
// ------
var rs_vi = {};
var trs_vi = {};
var tvs_vi = {};
var pas_vi = {};

var initGrid_vi = function() {
    var d3elt = d3.select('#draw_vi');
    d3elt.html('');

    var gh = env_vi.gh; // height in cells
    var gw = env_vi.gw; // width in cells
    var gs = env_vi.gs; // total number of cells

    var w = 420;
    var h = 420;
    svg = d3elt.append('svg').attr('width', w).attr('height', h)
      .append('g').attr('transform', 'scale(1)');

    // define a marker for drawing arrowheads
    svg.append("defs").append("marker")
    .attr("id", "arrowhead")
    .attr("refX", 3)
    .attr("refY", 2)
    .attr("markerWidth", 3)
    .attr("markerHeight", 4)
    .attr("orient", "auto")
    .append("path").attr("d", "M 0,0 V 4 L3,2 Z");

    for (var y=0; y<gh; y++) {
        for (var x=0; x<gw; x++) {
            var xcoord = x * cs;
            var ycoord = y * cs;
            var s = env_vi.xytos(x, y);
            var g = svg.append('g');

            // set up cell rectangles
            var r = g.append('rect')
            .attr('x', xcoord)
            .attr('y', ycoord)
            .attr('height', cs)
            .attr('width', cs)
            .attr('fill', '#FFF')
            .attr('stroke', 'black')
            .attr('stroke-width', 2);
            rs_vi[s] = r;

            // reward text
            var tr = g.append('text')
            .attr('x', xcoord + 5)
            .attr('y', ycoord + 55)
            .attr('font-size', 10)
            .text('');
            trs_vi[s] = tr;

            // skip rest for cliffs
            if (env_vi.T[s] === 1) { continue; }

            // value text
            var tv = g.append('text')
                .attr('x', xcoord + 5)
                .attr('y', ycoord + 20)
                .text('');
            tvs_vi[s] = tv;

            // policy arrows
            pas_vi[s] = []
            for (var a=0;a<4;a++) {
                var pa = g.append('line');
                pa.attr('x1', xcoord)
                  .attr('y1', ycoord)
                  .attr('x2', xcoord)
                  .attr('y2', ycoord)
                  .attr('stroke', 'black')
                  .attr('stroke-width', '2')
                  .attr("marker-end", "url(#arrowhead)");
                pa.attr('visibility', 'hidden');
                pas_vi[s].push(pa);
            }
        }
    }
}

var drawGrid_vi = function() {
    var gh = env_vi.gh; // height in cells
    var gw = env_vi.gw; // width in cells
    var gs = env_vi.gs; // total number of cells

    // updates the grid with current state of world/agent_vi
    for (var y=0; y<gh; y++) {
        for (var x=0; x<gw; x++) {
            var xcoord = x * cs;
            var ycoord = y * cs;
            var r=255, g=255, b=255;
            var s = env_vi.xytos(x, y);

            var vv = agent_vi.V[s];
            var ms = 100;
            if (vv > 0) { g = 255; r = 255 - vv * ms; b = 255 - vv * ms; }
            if (vv < 0) { g = 255 + vv * ms; r = 255; b = 255 + vv * ms; }
            var vcol = 'rgb(' + Math.floor(r) + ',' + Math.floor(g) + ',' + Math.floor(b) + ')';
            if (env_vi.T[s] === 1) { vcol = "#AAA"; rcol = "#AAA"; }

            // update colors of rectangles based on value
            var r = rs_vi[s];
            var rv = env_vi.Rarr[s];

            if (rv === 5) {
                // highlight goal cell
                r.attr('fill', '#FF0');
            } else if (rv == -5) {
                // highlight cliff cell
                r.attr('fill', '#aaa');
            } else {
                r.attr('fill', vcol);
            }

            // write reward texts
            trs_vi[s].text('R ' + rv.toFixed(1))

            // skip rest for cliff
            if (env_vi.T[s] === 1) continue;

            // write value
            if(s !== 16 && s !== 32) {
                var tv = tvs_vi[s];
                tv.text(agent_vi.V[s].toFixed(2));
            }
        }
    }
}

var drawArrow_vi = function() {
    var gh = env_vi.gh; // height in cells
    var gw = env_vi.gw; // width in cells
    var gs = env_vi.gs; // total number of cells

    // updates the grid with current state of world/agent_vi
    for (var y=0; y<gh; y++) {
        for (var x=0; x<gw; x++) {
            var xcoord = x * cs;
            var ycoord = y * cs;
            var s = env_vi.xytos(x, y);
            if(s !== 16 && s !== 32) {
                var paa = pas_vi[s];
                for (var a = 0; a < 4; a++) {
                    var pa = paa[a];
                    var prob = agent_vi.P[a * gs + s];

                    var status = $('#vi_status').text();
                    if (status == 'STOP') {
                        pa.attr('visibility', 'visible');
                    }

                    var ss = cs / 2 * prob * 0.9;
                    if (a === 0) {
                        nx = -ss;
                        ny = 0;
                    }
                    if (a === 1) {
                        nx = 0;
                        ny = -ss;
                    }
                    if (a === 2) {
                        nx = 0;
                        ny = ss;
                    }
                    if (a === 3) {
                        nx = ss;
                        ny = 0;
                    }

                    pa.attr('x1', xcoord + cs / 2)
                        .attr('y1', ycoord + cs / 2)
                        .attr('x2', xcoord + cs / 2 + nx)
                        .attr('y2', ycoord + cs / 2 + ny);

                    if (xcoord + cs / 2 == xcoord + cs / 2 + nx && ycoord + cs / 2 == ycoord + cs / 2 + ny) {
                        pa.attr('visibility', 'hidden');
                    } else {
                        pa.attr('visibility', 'visible');
                    }
                }
            }
        }
    }
}

var sid_vi = -1;
var vi_diff_state_values = 0;

var toggleValueIteration = function() {
    var num_iter = 1;
    if(sid_vi === -1) {
        sid_vi = setInterval(function(){
            $('#vi_status').text("START");
            agent_vi.evaluatePolicy();
            agent_vi.updatePolicy();

            drawGrid_vi();

            $('#vi_iter').text(num_iter);

            vi_diff_state_values = Math.abs(vi_diff_state_values - getViSumStateValues());
            $('#vi_state_values').text(getViSumStateValues().toFixed(1));
            if (vi_diff_state_values > 0.0 && vi_diff_state_values < 0.1) {
                stopValueIteration();
                $('#vi_diff_state_values').text(0.0.toFixed(1));
            } else {
                $('#vi_diff_state_values').text(vi_diff_state_values.toFixed(1));
                vi_diff_state_values = getViSumStateValues();
            }
            num_iter += 1;
        }, 500);
    } else {
        clearInterval(sid_vi);
        sid_vi = -1;
        $('#vi_status').text("PAUSE");
    }
}

function stopValueIteration() {
    $('#vi_status').text("STOP");
    drawArrow_vi();
    env_vi.reset();
    agent_vi.reset();
    clearInterval(sid_vi);
    sid_vi = -1;
    $('#vi_toggle').prop('disabled', true);
    $('#vi_stop').prop('disabled', true);
}

function resetValueIteration() {
    env_vi.reset();
    agent_vi.reset();
    clearInterval(sid_vi);
    $('#vi_iter').text(0);
    initGrid_vi();
    drawGrid_vi();
    sid_vi = -1;
    $('#vi_diff_state_values').text(-1);
    $('#vi_state_values').text(-1);
    $('#vi_status').text("RESET");
    $('#vi_toggle').prop('disabled', false);
    $('#vi_stop').prop('disabled', false);
}

function getViSumStateValues() {
    var sumStateValues = 0;
    for (var i=0; i < agent_vi.ns; i++) {
        sumStateValues += agent_vi.V[i];
    }
    return sumStateValues;
}

function init_vi() {
    env_vi = new gridword_vi(); // create environment
    agent_vi = new RL_VI.DPAgent(env_vi, {'gamma':0.9}); // create an agent_vi, yay!
    initGrid_vi();
    drawGrid_vi();
}
























