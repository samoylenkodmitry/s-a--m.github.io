---
layout: leetcode-entry
title: "841. Keys and Rooms"
permalink: "/leetcode/problem/2022-12-20-841-keys-and-rooms/"
leetcode_ui: true
entry_slug: "2022-12-20-841-keys-and-rooms"
---

[841. Keys and Rooms](https://leetcode.com/problems/keys-and-rooms/description/) medium

[https://t.me/leetcode_daily_unstoppable/58](https://t.me/leetcode_daily_unstoppable/58)

[blog post](https://leetcode.com/problems/keys-and-rooms/solutions/2932740/kotlin-dfs/)

```kotlin
    fun canVisitAllRooms(rooms: List<List<Int>>): Boolean {
       val visited = hashSetOf(0)
       with(ArrayDeque<Int>()) {
           add(0)
           while(isNotEmpty()) {
               rooms[poll()].forEach {
                   if (visited.add(it)) add(it)
               }
           }
       }
       return visited.size == rooms.size
    }

```

We need to visit each room, and we have positions of the other rooms and a start position. This is a DFS problem.
Keep all visited rooms numbers in a hash set and check the final size. Other solution is to use boolean array and a counter of the visited rooms.

Space: O(N) - for queue and visited set, Time: O(N) - visit all the rooms once

