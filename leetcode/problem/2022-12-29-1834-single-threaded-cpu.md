---
layout: leetcode-entry
title: "1834. Single-Threaded CPU"
permalink: "/leetcode/problem/2022-12-29-1834-single-threaded-cpu/"
leetcode_ui: true
entry_slug: "2022-12-29-1834-single-threaded-cpu"
---

[1834. Single-Threaded CPU](https://leetcode.com/problems/single-threaded-cpu/description/) medium

[https://t.me/leetcode_daily_unstoppable/67](https://t.me/leetcode_daily_unstoppable/67)

[blog post](https://leetcode.com/problems/single-threaded-cpu/solutions/2966855/kotlin-two-heaps/)

```kotlin
    fun getOrder(tasks: Array<IntArray>): IntArray {
        val pqSource = PriorityQueue<Int>(compareBy(
            { tasks[it][0] },
            { tasks[it][1] },
            { it }
        ))
        (0..tasks.lastIndex).forEach { pqSource.add(it) }
        val pq = PriorityQueue<Int>(compareBy(
            { tasks[it][1] },
            { it }
        ))
        val res = IntArray(tasks.size) { 0 }
        var time = 1
        for(resPos in 0..tasks.lastIndex) {
            while (pqSource.isNotEmpty() && tasks[pqSource.peek()][0] <= time) {
                pq.add(pqSource.poll())
            }
            if (pq.isEmpty()) {
                //idle
                pq.add(pqSource.poll())
                time = tasks[pq.peek()][0]
            }
            //take task
            val taskInd = pq.poll()
            val task = tasks[taskInd]
            time += task[1]
            res[resPos] = taskInd
        }
        return res
    }

```

First we need to sort tasks by their availability (and other rules),
then take tasks one by one and add them to another sorted set/heap where their start time doesn't matter,
but running time and order does. When we take the task from the heap, we increase the time and fill in the heap.
* use two heaps, one for the source of tasks, another for the current available tasks.
* don't forget to increase time to the nearest task if all of them unavailable

Space: O(n), Time: O(nlogn)

