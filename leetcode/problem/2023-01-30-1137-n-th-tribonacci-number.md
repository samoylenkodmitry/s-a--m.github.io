---
layout: leetcode-entry
title: "1137. N-th Tribonacci Number"
permalink: "/leetcode/problem/2023-01-30-1137-n-th-tribonacci-number/"
leetcode_ui: true
entry_slug: "2023-01-30-1137-n-th-tribonacci-number"
---

[1137. N-th Tribonacci Number](https://leetcode.com/problems/n-th-tribonacci-number/description/) easy

[blog post](https://leetcode.com/problems/n-th-tribonacci-number/solutions/3116945/kotlin-code-golf/)

```kotlin
    fun tribonacci(n: Int): Int = if (n < 2) n else {
        var t0 = 0
        var t1 = 1
        var t2 = 1
        repeat(n - 2) {
            t2 += (t0 + t1).also {
                t0 = t1
                t1 = t2
            }
        }
        t2
    }

```

#### Telegram
https://t.me/leetcode_daily_unstoppable/102
#### Intuition
Just do what is asked.
#### Approach
* another way is to use dp cache
#### Complexity
- Time complexity:
  $$O(n)$$
- Space complexity:
  $$O(1)$$

