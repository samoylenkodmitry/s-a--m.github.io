---
layout: post
title: How to debug Android apps over the Internet
---

# How to debug Android apps over the Internet

## The problem

You working remotely and you and the target device are too far away. Or you work on a remote server and want to debug your app on your phone.

## The solution

1. Connect phone to the one machine with adb, let's call it "machine A
2. On the machine A, run `adb kill-server` and `adb start-server`
3. On the machine A, run `ssh -R 5555:localhost:5555 machineB`
3. On the remote machine, where you want to debug, run `adb devices` to check that the phone is connected