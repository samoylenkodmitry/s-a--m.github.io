---
layout: post
title: How to cast Linux audio and microphone to Discord
---
# How to cast Linux audio and microphone to Discord

## The problem

Discord on Linux doesn't have the option by defaul to play audio from you apps simultaneously with your microphone.

## The solution

What we do is to create two virtual sinks, one for microphone and another for music. 

```bash

pactl load-module module-null-sink sink_name=MicSink sink_properties=device.description=MicSink
pactl load-module module-null-sink sink_name=MusicSink sink_properties=device.description=MusicSink

```

Thats just two empty devices, let's bring them to life.
We want to play music in our output device (in my case it is called "TU106")

```bash

pactl load-module module-loopback source="MusicSink.monitor" sink="TU106"

```

And we also want to send the same audio to the microphone sink, as it will go to Discord.

```bash

pactl load-module module-loopback source="MusicSink.monitor" sink="MicSink"

```

The final step is to send your physical microphone to the virtual microphone sink. (in my case mic called "NoiseTorch Mic...")

```bash

pactl load-module module-loopback source="NoiseTorch Microphone for THRONMAX PULSE MICROPHONE" sink=MicSink

```

## The final setup

Now, go to pulseaudio pavucontrol and in Playback select "MusicSink" for your music player, and in Recording select "MicSink" for Discord. One of the "loopback-..." devices must play in your physical output device, others go to MicSink.