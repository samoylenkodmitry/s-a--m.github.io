---
layout: leetcode-entry
title: "1472. Design Browser History"
permalink: "/leetcode/problem/2023-03-18-1472-design-browser-history/"
leetcode_ui: true
entry_slug: "2023-03-18-1472-design-browser-history"
---

[1472. Design Browser History](https://leetcode.com/problems/design-browser-history/description/) medium

[blog post](https://leetcode.com/problems/design-browser-history/solutions/3310280/kotlin-list/)

```kotlin

class BrowserHistory(homepage: String) {
    val list = mutableListOf(homepage)
    var curr = 0
    var last = 0

    fun visit(url: String) {
        curr++
        if (curr == list.size) {
            list.add(url)
        } else {
            list[curr] = url
        }
        last = curr
    }

    fun back(steps: Int): String {
        curr = (curr - steps).coerceIn(0, last)
        return list[curr]
    }

    fun forward(steps: Int): String {
        curr = (curr + steps).coerceIn(0, last)
        return list[curr]
    }

}

```

#### Join me on telegram
https://t.me/leetcode_daily_unstoppable/152
#### Intuition
Simple solution with array list will work, just not very optimal for the memory.

#### Approach
Just implement it.
#### Complexity
- Time complexity:
$$O(1)$$ for all operations
- Space complexity:
$$O(n)$$, will keep all the links

