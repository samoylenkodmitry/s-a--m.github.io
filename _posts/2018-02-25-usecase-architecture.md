---
layout: post
title: Use Case Architecture for Android Applications
---

# Background
In the Android community, many different architectural approaches have formed. We started with Activity, then Fragments were added, and we needed to organize all this in a way that didn't turn into God objects supported by crutches over time, but was convenient for both expansion and modification. Developers scratched their heads and remembered that MVC, MVP, and other abbreviations were already invented for them. Currently, the most popular architectures are presented on the page https://github.com/googlesamples/android-architecture.

# What is Architecture?
It's not just UI.

# One Activity or Several?
I think - one. If the application has only one activity, we can say for sure that the visual interaction with the user occurs from the moment of onCreate to the moment of onDestroy (with breaks for onStop-onStart when minimized). At the very least, this is convenient.

# What Google Offers

https://developer.android.com/topic/libraries/architecture/index.html

Activity has a lifecycle. In Google's new Architecture Components, this lifecycle has been extended to data, resulting in LiveData. Any object can now be complicated with a lifecycle using the LifecycleOwner interface of LifecycleActivity and LifecycleFragment components. Subsequently, these classes were declared @Deprecated, as inheriting someone else's class is not convenient if there is already an established hierarchy.

# Reactive Approach

What if we consider the application as a set of events and reactions to them?

Indeed, what is Activity? It's a class-entry point. It has event-events, telling that the user saw the visual part and that the visual part is hidden. There are also some additional callbacks, like a new incoming intent, through which you can understand where the launch came from. Or pressing physical volume buttons. You can also catch the result of launching another activity. That's all, in the rest of the application we use only this basic information.

Let's now imagine that we can do without activity. Let's separate user events into a separate stream of events. BehaviorSubject is well suited for this. It will store only the last state.


```
	BehaviorSubject<StateEvent> getStateEventBehaviorSubject(final int type) {
		BehaviorSubject<StateEvent> eventsObservable = mEventObservablesByType.get(type);
		if (eventsObservable == null) {
			eventsObservable = BehaviorSubject.create();
			mEventObservablesByType.put(type, eventsObservable);
		}
		return eventsObservable;
	}
```
Now you can subscribe to lifecycle events:

```
appStatesGraph
				.eventsOfType(AppStatesGraph.Type.LIFECYCLE, LifecycleEventResume.class)
        .subscribe()
```
Other application states are turned into streams of state change events. For example, the state of having or not having an internet connection will be represented by two events:

```
public class Connected extends SimpleEvent {
	
	public Connected() {
		super(AppStatesGraph.Type.CONNECTION);
	}
}
public class Disconnected extends SimpleEvent {
	
	public Disconnected() {
		super(AppStatesGraph.Type.CONNECTION);
	}
}
```
And we can introduce non-trivial logic based on a combination of states. For example, here's the start of the application only with internet and after initial initialization:


```
appStatesGraph
				.eventsOfType(AppStatesGraph.Type.APP_KEYS_INITIALIZED)
				.take(1)
				
				.zipWith(appStatesGraph
					.eventsOfType(AppStatesGraph.Type.LIFECYCLE, LifecycleEventStart.class)
          .take(1), (o1, o2)
					-> RxUtils.IGNORED)
				
				.zipWith(appStatesGraph
					.eventsOfType(AppStatesGraph.Type.CONNECTION, Connected.class)
          .take(1), (o1, o2)
					-> RxUtils.IGNORED)
         
         .subscribe(); //todo стартуем, когда все условия соблюдены
```

Such combinations will be called UseCases. The convenience of them lies in the fact that the action logic is moved to a separate UseCase entity, which can be covered with tests. It does not require dependence on Activity or any other SDK component that requires non-trivial mock in the test. Another plus is encapsulation - all the rules of this UseCase are in a separate and only class. But in it, there should be no other rules not related to the current UseCase, and you need to follow this (I see this as a minus of the approach). Name the class with a name that contains information about what_does and when_does this particular UseCase do, and adhere to strict restrictions.

For example, in the current project with this approach, there are such UseCase classes:

```
UseCaseAppCheckWhoAmIOnStart
UseCaseShowOnboardingOnGuideEndOrSkip
UseCaseShowMainPageAfterOnboardings
UseCaseShowDialogsOnAppStart
...

These classes subscribe to streams and do exactly one thing, which is reflected in their names.

# Conclusion

The article presents just an approach. The specific implementation can be any at your discretion. For example, I liked how it can be done using separate classes, connected with the help of Dagger. This allowed creating the central part of the application with business logic, located on a higher level than just the logic of one screen. Usually, this logic is found in parts associated with entry-point classes, such as Activity, Fragments, Receivers. 

Consider this approach if the logic of your application goes beyond the context of one screen.





