### 모스키토 설치
<code>brew install mosquitto</code>
### 모스키토 서비스 실행
<code>brew services start mosquitto</code>
### 모스키토 서비스 중지
<code>brew services stop mosquitto</code>
### TEST
1) Subscribing to a topic:

<code>mosquitto_sub -d -h localhost -p 1883 -t "myfirst/test"</code>

2) Other client publishes a message content to that topic:

<code>mosquitto_pub -d -h localhost -p 1883 -t "myfirst/test" -m "Hello"</code>

## Reference
- http://lemonheim.blogspot.com/2017/01/mqtt-mosquitto-mac.html
- https://stackoverflow.com/questions/30823894/iot-mosquitto-mqtt-how-to-test-on-localhost#comment94350150_30823894