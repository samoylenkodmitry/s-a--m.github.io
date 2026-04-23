---
layout: leetcode-entry
title: "343. Integer Break"
permalink: "/leetcode/problem/2023-10-06-343-integer-break/"
leetcode_ui: true
entry_slug: "2023-10-06-343-integer-break"
---

[343. Integer Break](https://leetcode.com/problems/integer-break/description/) medium
[blog post](https://leetcode.com/problems/integer-break/solutions/4136139/kotlin-dfs-memo/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/6102023-343-integer-break?r=2bam17&utm_campaign=post&utm_medium=web)
![image.png](/assets/leetcode_daily_images/e910057a.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/361

#### Problem TLDR

Max multiplication of the number split

#### Intuition

We can search from all possible splits. The result will only depend on the input `n`, so can be cached.

#### Approach

* one corner case is the small numbers, like `2, 3, 4`: ensure there is at least one split happen

#### Complexity

- Time complexity:
$$O(n^2)$$, recursion depth is `n` and another `n` is in the loop. Without cache, it would be n^n

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    val cache = mutableMapOf<Int, Int>()
    fun integerBreak(n: Int, canTake: Boolean = false): Int =
      if (n == 0) 1 else cache.getOrPut(n) {
        (1..if (canTake) n else n - 1).map {
          it * integerBreak(n - it, true)
        }.max()
      }

```

