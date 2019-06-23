# -*- coding:utf-8 -*-
import paho.mqtt.client as mqtt


# 서버로부터 CONNTACK 응답을 받을 때 호출되는 콜백
def on_connect(client, userdata, flags, rc):
    print("Connected with result code {}".format(rc))
    client.subscribe("episode") # 구독 "episode"


# 서버로부터 publish message를 받을 때 호출되는 콜백
def on_message(client, userdata, msg):
    print(msg.topic + " " + str(msg.payload))


client = mqtt.Client()          # client 오브젝트 생성
client.on_connect = on_connect  # 콜백설정
client.on_message = on_message  # 콜백설정

client.connect("localhost", 1883, 60)
client.loop_forever()
