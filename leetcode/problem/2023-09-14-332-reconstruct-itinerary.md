---
layout: leetcode-entry
title: "332. Reconstruct Itinerary"
permalink: "/leetcode/problem/2023-09-14-332-reconstruct-itinerary/"
leetcode_ui: true
entry_slug: "2023-09-14-332-reconstruct-itinerary"
---

[332. Reconstruct Itinerary](https://leetcode.com/problems/reconstruct-itinerary/description/) hard
[blog post](https://leetcode.com/problems/reconstruct-itinerary/solutions/4042335/kotlin-dfs-backtrack/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/14092023-332-reconstruct-itinerary?r=2bam17&utm_campaign=post&utm_medium=web)
![image.png](/assets/leetcode_daily_images/7cb18c6d.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/339

#### Problem TLDR

Smallest lexical order path using all the tickets

#### Intuition

We can build a graph, then do DFS in a lexical order, backtracking. First path with all tickets used will be the answer.

#### Approach

* graph has directed nodes
* sort nodes lists by strings comparison
* current node is always the last in the path

#### Complexity

- Time complexity:
$$O(x^n)$$, where x - is an average edges count per node

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun findItinerary(tickets: List<List<String>>): List<String> {
      val fromTo = mutableMapOf<String, MutableList<Pair<Int, String>>>()
      tickets.forEachIndexed { i, (from, to) ->
        fromTo.getOrPut(from) { mutableListOf() } += i to to
      }
      for (list in fromTo.values) list.sortWith(compareBy { it.second })
      val usedTickets = mutableSetOf<Int>()
      var path = mutableListOf("JFK")
      fun dfs(): List<String> =
        if (usedTickets.size == tickets.size) path.toList()
        else fromTo[path.last()]?.asSequence()?.map { (ind, next) ->
          if (usedTickets.add(ind)) {
            path.add(next)
            dfs().also {
              path.removeAt(path.lastIndex)
              usedTickets.remove(ind)
            }
          } else emptyList()
        }?.filter { it.isNotEmpty() }?.firstOrNull() ?: emptyList()
      return dfs()
    }

```

