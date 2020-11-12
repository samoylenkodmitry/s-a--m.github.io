---
layout: post
title: Which is faster? ConcurrentSkipListSet vs TreeSet+synchronized
---
# What's the difference
Those two collections are all sorted and very useful when you need to maintain some order in dynamically filled data.
However when concurrency takes place there is a gotcha. We have a choice: use manual synchronization or use ConcurrentSkipListSet from java.concurrent package.

# My case
I have a task that gets episodes from server in a bulk operation. For example, when there is a 800 episodes total it makes 8 asynchronious requests 100 episodes each.
Resulting list of episodes needs to be sorted and displayed on the client side.

Code for manual synchronizations looks like this:
```
final Set<Video> allEpisodes = new TreeSet<>(sortByEpisode);

final Video[] sortedEpisodes;

synchronized (allEpisodes) {
	Collections.addAll(allEpisodes, notFakeVideos);
	sortedEpisodes = allEpisodes.toArray(Video.EMPTY_ARRAY);
}
//use of sortedEpisodes
```
which is somewhat verbose.

In the opposite, code with concurrent collection looks very clean.:

```
final Set<Video> allEpisodes = new ConcurrentSkipListSet<>(sortByEpisode);

Collections.addAll(allEpisodes, notFakeVideos);

final Video[] sortedEpisodes = allEpisodes.toArray(Video.EMPTY_ARRAY);

```
Almost none of the concurrent stuff is visible to programmer which is a good thing for maintainability and error-proneness.
				
# Benchmark?

I'm tested two solutions using Android emulator API 30. Here is a results:

solution | avg ns | min ns | max ns | # of experiments | sum ns
--- | --- | --- | --- | --- | ---
concurrent | 114960 | 6543 | 4109831 | 620 | 71275591
synchronized | 44295 | 7035 | 1525028 | 620 | 27463261

For my use case manual synchronization is almost twice as fast as using concurrent collection.

# But which is better?
Winner: TreeSet+synchronized!
However in broad case answer is: "It depends". 
My case consist of only 10 invokations of adding elements and one `toArray()` call. It is particulary small number of concurrent events. 
That's why concurrent collection overhead makes visible difference for performance.
