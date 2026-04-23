---
layout: leetcode-entry
title: "Guess Number Higher Or Lower"
permalink: "/leetcode/problem/2022-11-16-guess-number-higher-or-lower/"
leetcode_ui: true
entry_slug: "2022-11-16-guess-number-higher-or-lower"
---

[https://leetcode.com/problems/guess-number-higher-or-lower/](https://leetcode.com/problems/guess-number-higher-or-lower/) easy

```kotlin

    override fun guessNumber(n:Int):Int {
       var lo = 1
       var hi = n
       while(lo <= hi) {
           val pick = lo + (hi - lo)/2
           val answer = guess(pick)
           if (answer == 0) return pick
           if (answer == -1) hi = pick - 1
           else lo = pick + 1
       }
       return lo
    }

```

This is a classic binary search algorithm.
The best way of writing it is:
* use safe mid calculation (lo + (hi - lo)/2)
* use lo <= hi instead of lo < hi and mid+1/mid-1 instead of mid

Complexity: O(log(N))
Memory: O(1)

