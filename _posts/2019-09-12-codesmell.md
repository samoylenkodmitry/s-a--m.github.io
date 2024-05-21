---
layout: post
title: Basic Guidelines for Writing Code
---

# Basic Guidelines for Writing Code

#### 1. The code should be as clear as possible. If someone doesn't understand your code during a review, it means the code is not clear enough. If someone has edited your code, refactored it, and made a mistake - the code is not clear enough.
#### 2. The code should be simple. Simple code always does one thing. If it’s a method - it performs one action. If it’s a class - it is responsible for one area. If it’s a module - it contains only what is necessary and relevant to it.
#### 3. The code should be optimally performant. This means:

  * not creating unnecessary objects for GC
  * not performing waiting IO actions where they can be executed later
  * not filling up memory (e.g., by creating many threads)
  * not allowing objects to be retained beyond their lifecycle (memory leaks), deinitialize what is no longer needed
  * knowing the average time of each operation you write (e.g., how much it costs to call new Error(), new Thread(), etc.)
  * read the book Effective Java
  
#### 4. The code should not contain concurrency errors.

  * Always monitor the lifecycle of objects
  * Concurrency can also occur in single-threaded code executed in Looper.loop()
  * Always assume that the code is multi-threaded. A field you write/read to may be null. Text that is assembled in StringBuilder and read from different threads can turn into garbage
  * read the book Concurrency in Practice
  
#### 5. The code should not crash in production.

  * Write unit tests for places where you have doubts. (Pro-level: also write unit tests for places where you don't have doubts)
  * If you are not sure about yourself (or a third-party library), wrap everything in try..catch + send a non-fatal report to your crash reporting service
  
#### 6. The code should never crash in production.

  * If an error occurs due to a changed server format - it's the client's fault. The code should be able to not crash, but report the problem.
  * If an error occurs once a year due to the phase of the moon against the backdrop of the Sagittarius constellation - it's the client's fault.
  * There should be no uncaught Exceptions in working code.
  
#### 7. You should enjoy your code. If you don't feel this, try to rest, take a vacation. If you have never felt this, you should expand your horizons. Challenge yourself, participate in codegolf, aicup, solve problems on hackerrank, leetcode. Try to write something for yourself personally on your phone.




