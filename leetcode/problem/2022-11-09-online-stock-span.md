---
layout: leetcode-entry
title: "Online Stock Span"
permalink: "/leetcode/problem/2022-11-09-online-stock-span/"
leetcode_ui: true
entry_slug: "2022-11-09-online-stock-span"
---

[https://leetcode.com/problems/online-stock-span/](https://leetcode.com/problems/online-stock-span/) medium

So, we need to keep increasing sequence of numbers, increasing/decreasing stack will help.
Consider example, this is how decreasing stack will work
```bash
        // 100   [100-1]                            1
        // 80    [100-1, 80-1]                      1
        // 60    [100-1, 80-1, 60-1]                1
        // 70    [100-1, 80-1, 70-2] + 60           2
        // 60    [100-1, 80-1, 70-2, 60-1]          1
        // 75    [100-1, 80-1, 75-4] + 70-2+60-1    4
        // 85    [100-1, 85-6] 80-1+75-4            6

```

Solution:

```kotlin

class StockSpanner() {
    val stack = Stack<Pair<Int,Int>>()

    fun next(price: Int): Int {
        // 100   [100-1]                            1
        // 80    [100-1, 80-1]                      1
        // 60    [100-1, 80-1, 60-1]                1
        // 70    [100-1, 80-1, 70-2] + 60           2
        // 60    [100-1, 80-1, 70-2, 60-1]          1
        // 75    [100-1, 80-1, 75-4] + 70-2+60-1    4
        // 85    [100-1, 85-6] 80-1+75-4            6
       var span = 1
       while(stack.isNotEmpty() && stack.peek().first <= price) {
          span += stack.pop().second
       }
       stack.push(price to span)
       return span
    }

}

```

Complexity: O(N)
Memory: O(N)

