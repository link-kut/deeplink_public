var RL_PI = {};

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

  // exports
  global.DPAgent = DPAgent;
})(RL_PI);

var agent_pi, env_pi;

// gridword_pi
var gridword_pi = function(){
    this.Rarr = null; // reward array
    this.T = null; // cell types, 0 = normal, 1 = cliff
    this.reset()
};

gridword_pi.prototype = {
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
var rs_pi = {};
var trs_pi = {};
var tvs_pi = {};
var pas_pi = {};

var initGrid_pi = function() {
    var d3elt = d3.select('#draw_pi');
    d3elt.html('');

    var gh = env_pi.gh; // height in cells
    var gw = env_pi.gw; // width in cells
    var gs = env_pi.gs; // total number of cells

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
            var s = env_pi.xytos(x, y);
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
            rs_pi[s] = r;

            // reward text
            var tr = g.append('text')
            .attr('x', xcoord + 5)
            .attr('y', ycoord + 55)
            .attr('font-size', 10)
            .text('');
            trs_pi[s] = tr;

            // skip rest for cliffs
            if (env_pi.T[s] === 1) { continue; }

            // value text
            var tv = g.append('text')
            .attr('x', xcoord + 5)
            .attr('y', ycoord + 20)
            .text('');
            tvs_pi[s] = tv;

            // policy arrows
            pas_pi[s] = []
            for (var a=0;a<4;a++) {
                var pa = g.append('line')
                  .attr('x1', xcoord)
                  .attr('y1', ycoord)
                  .attr('x2', xcoord)
                  .attr('y2', ycoord)
                  .attr('stroke', 'black')
                  .attr('stroke-width', '2')
                  .attr("marker-end", "url(#arrowhead)");
                pas_pi[s].push(pa);
            }
        }
    }
}

var drawGrid_pi = function() {
    var gh = env_pi.gh; // height in cells
    var gw = env_pi.gw; // width in cells
    var gs = env_pi.gs; // total number of cells

    // updates the grid with current state of world/agent_pi
    for (var y=0; y<gh; y++) {
        for (var x=0; x<gw; x++) {
            var xcoord = x * cs;
            var ycoord = y * cs;
            var r=255, g=255, b=255;
            var s = env_pi.xytos(x, y);

            var vv = agent_pi.V[s];
            var ms = 100;
            if (vv > 0) { g = 255; r = 255 - vv * ms; b = 255 - vv * ms; }
            if (vv < 0) { g = 255 + vv * ms; r = 255; b = 255 + vv * ms; }
            var vcol = 'rgb(' + Math.floor(r) + ',' + Math.floor(g) + ',' + Math.floor(b) + ')';
            if (env_pi.T[s] === 1) { vcol = "#AAA"; rcol = "#AAA"; }

            // update colors of rectangles based on value
            var r = rs_pi[s];
            var rv = env_pi.Rarr[s];

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
            trs_pi[s].text('R ' + rv.toFixed(1))

            // skip rest for cliff
            if (env_pi.T[s] === 1) continue;

            // write value
            if(s !== 16 && s !== 32) {
                var tv = tvs_pi[s];
                tv.text(agent_pi.V[s].toFixed(2));

                // update policy arrows
                var paa = pas_pi[s];
                for (var a = 0; a < 4; a++) {
                    var pa = paa[a];
                    var prob = agent_pi.P[a * gs + s];

                    if (prob === 0) {
                        pa.attr('visibility', 'hidden');
                    }
                    else {
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
                }
            }
        }
    }
}

var sid_pi = -1;
var pi_diff_state_values = 0;

var togglePolicyIteration = function() {
    var num_iter = 1;
    if(sid_pi === -1) {
        sid_pi = setInterval(function(){
            $('#pi_status').text("START");
            agent_pi.evaluatePolicy();
            agent_pi.updatePolicy();
            drawGrid_pi();
            $('#pi_iter').text(num_iter);

            pi_diff_state_values = Math.abs(pi_diff_state_values - getPiSumStateValues());
            $('#pi_state_values').text(getPiSumStateValues().toFixed(1));
            if (pi_diff_state_values > 0.0 && pi_diff_state_values < 0.1) {
                stopPolicyIteration();
                $('#pi_diff_state_values').text(0.0.toFixed(1));
            } else {
                $('#pi_diff_state_values').text(pi_diff_state_values.toFixed(1));
                pi_diff_state_values = getPiSumStateValues();
            }
            num_iter += 1;
        }, 500);
    } else {
        clearInterval(sid_pi);
        sid_pi = -1;
        $('#pi_status').text("PAUSE");
    }
}

function stopPolicyIteration() {
    $('#pi_status').text("STOP");
    drawGrid_pi();
    env_pi.reset();
    agent_pi.reset();
    clearInterval(sid_pi);
    sid_pi = -1;
    $('#pi_toggle').prop('disabled', true);
    $('#pi_stop').prop('disabled', true);
}

function resetPolicyIteration() {
    env_pi.reset();
    agent_pi.reset();
    clearInterval(sid_pi);
    $('#pi_iter').text(0);
    initGrid_pi();
    drawGrid_pi();
    sid_pi = -1;
    $('#pi_diff_state_values').text(-1);
    $('#pi_state_values').text(-1);
    $('#pi_status').text("RESET");
    $('#pi_toggle').prop('disabled', false);
    $('#pi_stop').prop('disabled', false);
}

function getPiSumStateValues() {
    var sumStateValues = 0;
    for (var i=0; i < agent_pi.ns; i++) {
        sumStateValues += agent_pi.V[i];
    }
    return sumStateValues;
}

function init_pi() {
    env_pi = new gridword_pi(); // create environment
    agent_pi = new RL_PI.DPAgent(env_pi, {'gamma':0.9}); // create an agent_pi, yay!
    initGrid_pi();
    drawGrid_pi();
}

// function info() {
//     $('#ns').text(env_pi.getNumStates());
//     $('#na').text(env_pi.getMaxNumActions());
//     poss_action = []
//     for(var s=0; s<agent_pi.ns; s++) {
//         var poss = this.env_pi.allowedActions(s);
//         var poss_action_names = [];
//         for (var k=0; k<poss.length; k++) {
//             poss_action_names.push(action_names[poss[k]] + "(" + poss[k] + ")" + " - " + agent_pi.P[poss[k]*agent_pi.ns + s].toFixed(3));
//         }
//         poss_action.push(s + " - " + env_pi.stox(s) + ":" + env_pi.stoy(s) + " - " + poss_action_names + "<br/>");
//     }
//     $('#tp').html(poss_action);
// }