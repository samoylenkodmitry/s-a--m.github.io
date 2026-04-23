---
layout: leetcode-entry
title: "739. Daily Temperatures"
permalink: "/leetcode/problem/2022-12-18-739-daily-temperatures/"
leetcode_ui: true
entry_slug: "2022-12-18-739-daily-temperatures"
---

[739. Daily Temperatures](https://leetcode.com/problems/daily-temperatures/description/) medium

[https://t.me/leetcode_daily_unstoppable/55](https://t.me/leetcode_daily_unstoppable/55)

[blog post](https://leetcode.com/problems/daily-temperatures/solutions/2924323/kotlin-increasing-stack/)

```kotlin
    fun dailyTemperatures(temperatures: IntArray): IntArray {
       val stack = Stack<Int>()
       val res = IntArray(temperatures.size) { 0 }
       for (i in temperatures.lastIndex downTo 0) {
           while(stack.isNotEmpty() && temperatures[stack.peek()] <= temperatures[i]) stack.pop()
           if (stack.isNotEmpty()) {
               res[i] = stack.peek() - i
           }
           stack.push(i)
       }
       return res
    }

```

Intuitively, we want to go from the end of the array to the start and keep the maximum value. But, that doesn't work, because we must also store smaller numbers, as they are closer in distance.
For example, `4 3 5 6`, when we observe `4` we must compare it to `5`, not to `6`. So, we store not just max, but increasing max: `3 5 6`, and throw away all numbers smaller than current, `3 < 4` - pop().

We will iterate in reverse order, storing indexes in increasing by temperatures stack.

Space: O(N), Time: O(N)

