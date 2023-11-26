---
layout: post
title: Implementing a Global try-catch for an Android Application
---
# Where in the Application is the main() Method Hidden?

It can be found empirically. Set a breakpoint in the Application.onCreate callback and
you will discover it at the top of the stack trace, in the class ActivityThread.
Here it is, the "familiar" entry point of a JVM application:
```
    public static void main(String[] args) {
        ...
        Looper.prepareMainLooper();

        ActivityThread thread = new ActivityThread();
        thread.attach(false, startSeq);
        ...

        Looper.loop();

        throw new RuntimeException("Main thread loop unexpectedly exited");
    }

```
Skipping things irrelevant to our topic, here's what's happening:
1. `Looper.prepareMainLooper()` "prepares" the Looper (creates a thread-local instance of the Looper class)
2. `ActivityThread thread =...` An instance of ActivityThread is created
3. `thread.attach` The attach method of ActivityThread is executed (creates an instance of the application and calls the Application.onCreate callback)
4. `Looper.loop()` The looper starts. From this moment, the looper begins to poll its internal queue and execute
tasks from it. All other activity callbacks will end up here.
5. `throw new RuntimeException("Main thread loop unexpectedly exited")` The last line reveals
the essence of the SDK architecture - the application should not reach this line in its lifetime. If we do, 
the application terminates with an error.
Thus, a problem and its solution emerge: if some random Activity or View callback throws an error, the entire application crashes. There's no way to safeguard against this in advance.
In Java, there is a universal global mechanism for catching errors:

```
		Thread.currentThread().setUncaughtExceptionHandler((t, e) -> ... ); //ловим все ошибки
```
But the above architecture doesn't allow this mechanism to be used, as the error will first be thrown in the Looper.loop() call, ending it, and only then it will come out in main() and our set error handler.
Once the error is caught by the handler, we are left with the problem of the completed Looper.loop, and therefore
an application that no longer responds to system callbacks and clicks.

# Solution to the Frozen Application Problem
To get the activity to respond to clicks and callbacks again, just restart the looper:
```
		Looper.loop();
```

So, the final solution. In Application.onCreate:
```
		Thread.currentThread().setUncaughtExceptionHandler((t, e) -> continueSafeLoop(e));
```

```
	public static void continueSafeLoop(final Throwable e) {
		Throwable error = e;
		while (true) {
			Assert.fail(error);
			try {
				final Looper looper = Looper.myLooper();
				new Handler(looper).removeCallbacksAndMessages(null);
				final MessageQueue queue = ReflectUtils.readField(looper, "mQueue");
				final Long ptr = ReflectUtils.readField(queue, "mPtr");
				final Boolean quitting = ReflectUtils.readField(queue, "mQuitting");
				if ((ptr == null || ptr != 0) && (quitting == null || !quitting.booleanValue())) {
					Looper.loop();
				} else {
				break; //this is just detecting the end of the looper; done only reflectively
				}
			} catch (final Throwable err) {
				error = err;
			}
		}
	}
```
Now you can experiment by throwing an error in any of the callbacks and see that the application does not crash
and does not freeze.
Of course, you will still need to abstract away from direct activity callbacks, as there is a detect call to the super method.
And you must ensure sending caught crashes to Firebase/Crashlytics, as there will no longer be crashes
in the standard error reports in the Google Play console.

P.S.: It is noteworthy that the class ActivityThread is not a Thread and is not about the Android class
Activity, but rather about "activity" in the sense of "a set of actions," and creates not one activity, but the entire Application.
In essence, it acts as a delegate of all system callbacks of the application, responsible for its state transitions.
This is also stated in its JavaDoc:
```
/**
 * This manages the execution of the main thread in an
 * application process, scheduling and executing activities,
 * broadcasts, and other operations on it as the activity
 * manager requests.
 *
 * {@hide}
 */
```
A bit of an unfortunate name choice, in my opinion :)




