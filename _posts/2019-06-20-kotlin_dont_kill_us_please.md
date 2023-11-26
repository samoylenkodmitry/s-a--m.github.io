---
layout: post
title: Kotlin Don't Kill Us Please
---
# Kotlin Don't Kill Us Please

We all, whether we want to or not, love Kotlin.
But your app's user doesn't necessarily need to love Kotlin. 
What's far more important to them is that the app doesn't crash and performs its primary function - to work.
Even if the string you wanted to show in a tooltip is empty. Why should the app crash? We came to watch a movie! (we're in the business of showing videos)

So, if you're already writing in Kotlin, like all successful people, you might have noticed something like this:

```
kotlin.UninitializedPropertyAccessException: lateinit property mChatAdapter has not been initialized // что-то вроде вот этого 
```
We all love absolutely correct, error-free code, but it's time to admit to ourselves that such code doesn't exist (yet).

Wait, I have a solution. Let's make our application do as much as possible to perform its main function. And all kinds of checks, asserts, etc., let them be logged, sent somewhere, written down. 
Actually, this is how we wrote our application. If there's an error somewhere, we need to quickly catch it, send the log to Fabric, but not crash the app.

Okay, but what about Kotlin?

Here Proguard will help us. We already know how to cut out logs, remove unnecessary string concatenations https://www.guardsquare.com/en/products/proguard/manual/examples#logging

Now our enemy (in the release, of course, in debug and development process - a helper) is this class https://github.com/JetBrains/kotlin/blob/v1.3.40/libraries/stdlib/jvm/runtime/kotlin/jvm/internal/Intrinsics.java

Let's add it to Proguard!


```
-assumenosideeffects class kotlin.jvm.internal.Intrinsics {

	public static void checkNotNull(...);

	public static void throwNpe(...);

	public static void throwUninitializedProperty(...);

	public static void throwUninitializedPropertyAccessException(...);

	public static void throwAssert(...);

	public static void throwIllegalArgument(...);

	public static void throwIllegalArgument(...);

	public static void throwIllegalState(...);

	public static void throwIllegalState(...);

	public static void checkExpressionValueIsNotNull(...);

	public static void checkNotNullExpressionValue(...);

	public static void checkReturnedValueIsNotNull(...);

	public static void checkReturnedValueIsNotNull(...);

	public static void checkFieldIsNotNull(...);

	public static void checkParameterIsNotNull(...);

	public static void checkNotNullParameter(...);

	private static void throwParameterIsNullException(...);

	public static void throwUndefinedForReified(...);

	public static void throwUndefinedForReified(...);

	public static void reifiedOperationMarker(...);

	public static void needClassReification(...);

	public static void checkHasClass(...);

}
```

Enjoy 100% crash free!
