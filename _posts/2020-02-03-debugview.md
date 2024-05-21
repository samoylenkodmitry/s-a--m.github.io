---
layout: post
title: How to Debug Layouts in Android
---

# How to Debug Layouts in Android

Sometimes, layouts start showing magic: hiding controls at the wrong moment, incorrectly determining their sizes and positions. 
Usually, this happens when the code controlling the layout is overly complex. For example, when one part of the application tries to hide a view while another tries to show it. 
And these actions are performed at arbitrary moments in time.

As a rule, this indicates improper architecture, and the correct solution would be to rewrite the complex control into a redux model with a single source of truth.

But if you don't have time to rewrite a large part of the application, the following approach will help:
* create a class inheriting the view under investigation (in this example, FrameLayout) and override the setVisibility method.

```java
class DebugView @JvmOverloads constructor(
	context: Context, attrs: AttributeSet? = null, defStyleAttr: Int = 0
) : FrameLayout(context, attrs, defStyleAttr) {
	init {
		if (!BuildConfig.DEBUG) {
			Assert.fail("Don't forget to remove debug view from release app")
		}
	}

	override fun setVisibility(visibility: Int) {
		super.setVisibility(visibility)

		// this is example of how to use it for listening to view changes
		Error("debug! $visibility $tag ${getTag(R.id.adjust_height)}").printStackTrace()
	}
}
```
In the layout, replace the view under investigation with our DebugView:


```xml
	<DebugView
		android:layout_width="match_parent"
		android:layout_height="wrap_content"
		>

		<TextView
...
			  
```

Now, when someone in the code changes the visibility of the investigated view, it can be caught.
Done!
