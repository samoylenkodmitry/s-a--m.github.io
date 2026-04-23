---
layout: leetcode-entry
title: "Ugly Number"
permalink: "/leetcode/problem/2022-11-18-ugly-number/"
leetcode_ui: true
entry_slug: "2022-11-18-ugly-number"
---

[https://leetcode.com/problems/ugly-number/](https://leetcode.com/problems/ugly-number/) easy

```

    fun isUgly(n: Int): Boolean {
        if (n <= 0) return false
        var x = n
        while(x%2==0) x = x/2
        while(x%3==0) x = x/3
        while(x%5==0) x = x/5
        return x == 1
    }

```

There is also a clever math solution, but I don't understand it yet.

Complexity: O(log(n))
Memory: O(1)

