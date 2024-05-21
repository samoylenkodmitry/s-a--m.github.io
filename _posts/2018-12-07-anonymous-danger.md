---
layout: post
title: The Dangers of Anonymous Classes
---

# Never Underestimate the Danger of Everyday Things

Sometimes in everyday Android development, you encounter strange, if not mystical bugs.
The fragmentation of the platform and the differences in the devices themselves make themselves known.

This time, I was creating an animated Drawable. A Callback is involved in the Drawable update mechanism, within which you need to inform the View about the need for redrawing.

```
//MyView.java

new Drawable.Callback() {
 @Override
 public void invalidateDrawable(@NonNull final Drawable who) {
invalidate();// <-- here invalidate is called for the View in which the Drawable is located
 }
		
...
}
```

Inside Drawable, I call Drawable#invalidateSelf() on every tick of the animation, which triggers this callback.

```
//MyDrawable.java

 public void startAnimation() {
  final ValueAnimator valueAnimator = ValueAnimator.ofInt(0, 100);
  valueAnimator.addUpdateListener(animation ->	{
   ...
invalidateSelf();// <--- here Drawable "tells" View through Callback that it needs to be redrawn
  });
  valueAnimator.start();
 }
```

![triangle]({{ site.url }}/assets/triangle.gif)

It seemed the task was done, but curiosity and experience with Android prompted me to check how this 
works on other system versions. And, as it turned out, my doubts were not unfounded - on Android 6 the animation
did not work fully, stopping halfway or not starting at all!

![triangle_not_work]({{ site.url }}/assets/triangle_not_work.png)

After long searches and logging, the culprit was found - it was the Drawable. During the animation process, it somehow lost the callback:

```
//MyDrawable.java
valueAnimator.addUpdateListener(animation ->	{
 ...
 invalidateSelf();
 ...
Log.d("x", "" + getCallback());// <-- in the middle of the animation this method started returning null
});
```

Looking inside, I discovered a "reliable" way to avoid memory leaks from the Google team:
```
//android.graphics.drawable.Drawable.java
 public final void setCallback(@Nullable Callback cb) {
mCallback = cb != null ? new WeakReference<>(cb) : null; // <-- why worry about leaks, let GC take care of it!
 }
```

After some grieving, I decided there was nothing to do but to live with what we had. In my implementation, I set the callback as a lambda:

```
//MyView.java
mSplashDrawable.setCallback(new Drawable.Callback() { // <-- anonymous class
...
});
```
GC works with WeakReference counting references, and if it doesn't find a single hard reference, it cleans it up. 
The lambda itself is an anonymous class and contains a hard reference to the outer class. The chain looks like this
```
MyView -> Drawable -> WeakReference -> Callback
<------------------------------| // <-- anonymous class reference to the parent

```

Only WeakReference points to the Callback, so GC collected it.

I replaced the anonymous class with a field, and GC no longer cleaned up my WeakReference:

```
//MyView
 private final Drawable.Callback mCallback = new Drawable.Callback() {
  ...
 }		
```

![triangle]({{ site.url }}/assets/triangle.gif)

The conclusion is this:
be careful with anonymous classes and even more cautious with the Android SDK ༼ʘ̚ل͜ʘ̚༽





