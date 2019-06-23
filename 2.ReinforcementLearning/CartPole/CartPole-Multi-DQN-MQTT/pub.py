import paho.mqtt.publish as publish

msgs = \
[
    {
        'topic':"/seoul/yuokok",
        'payload':"multiple 1"
    },

    (
        "/seoul/yuokok",
        "multiple 2", 0, False
    ) 
]
publish.multiple(msgs, hostname="localhost")
#Topic /seoul/yuokok 에 문자값 multiple 1, multiple 2를 발행한다.