---
layout: post
title: Databinding Episode I; Hidden Danger
---
# Databinding Episode I; Hidden Danger

Sometimes in everyday Android development, you encounter strange, if not mystical bugs. (standard introduction)

Our QA noticed a peculiar bug that three of our developers couldn't see even when staring at the screen. 
When transitioning to a screen, a part of it quickly "blinked" and then was hidden. The tester had very good FPS :)
Catching the frame with the "blink" in the suggested video, I doubted my ability to see this world, at least compared to some people. Well, it needs fixing.

Digging into the layout, I found a seemingly harmless piece of XML:

```
				...
				<ru.ivi.uikit.UiKitGridLayout
				...
					android:visibility="@{authState.isUserAuthorized ? View.GONE : View.VISIBLE, default=gone}"
				...
```

This part of the layout was supposed to be hidden by default and then appear or not, depending on the authorization status.
But for some reason, by default, it was shown and then hidden as it should be for an authorized user.

It seems there is a mistake in this line. Maybe ", default=gone"? The documentation won't help us here; we need to delve into the workings of data-binding:


```
       UserAuthorizedState authState = mAuthState;
        int authStateIsUserAuthorizedViewGONEViewVISIBLE = 0;
        boolean authStateIsUserAuthorized = false;

        if ((dirtyFlags & 0x3L) != 0) {
                if (authState != null) {
                    // read authState.isUserAuthorized
                    authStateIsUserAuthorized = authState.isUserAuthorized;
                }
            if((dirtyFlags & 0x3L) != 0) {
                if(authStateIsUserAuthorized) {
                        dirtyFlags |= 0x8L;
                }
                else {
                        dirtyFlags |= 0x4L;
                }
            }
                // read authState.isUserAuthorized ? View.GONE : View.VISIBLE
                authStateIsUserAuthorizedViewGONEViewVISIBLE = ((authStateIsUserAuthorized) ? (android.view.View.GONE) : (android.view.View.VISIBLE));
        }
        // batch finished
        if ((dirtyFlags & 0x3L) != 0) {
            // api target 1

            this.motivationToRegistration.setVisibility(authStateIsUserAuthorizedViewGONEViewVISIBLE);
        }
```
Following the logic of this generated code, we see that authStateIsUserAuthorized defaults to false.
If binding hasn't occurred yet, i.e., authState==null, then the variable's value doesn't change. 
As a result, the variable authStateIsUserAuthorizedViewGONEViewVISIBLE contains View.VISIBLE.
What about what we specified in the code ", default=gone"? 

There's no such thing as default! Here's the documentation, and it's not there https://developer.android.com/topic/libraries/data-binding/expressions see for yourself!
It only exists for strings and only in one of the answers on StackOverflow https://stackoverflow.com/questions/39241191/error-with-default-value-in-databinding?rq=1

That's the situation. 

Now let's remove the word default from the XML and compare the generated code - nothing has changed.
Let's just pretend that default is our fantasy of what databinding should look like. 
For now, let's replace the code in the XML with something more reliable:

```
android:visibility="@{authState==null||authState.isUserAuthorized ? View.GONE : View.VISIBLE}"
```

Now our layout can't be seen with any super-vision.


Conclusion:
be careful with code inside XML and databinding, and even more cautious with the Android SDK ༼ʘ̚ل͜ʘ̚༽

UPD: default - works, but only as a parameter that will be substituted in the XML upon inflation. Then a binding with null data may be called and overwrite the default value. 




