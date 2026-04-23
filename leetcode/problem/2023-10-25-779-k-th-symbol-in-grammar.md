---
layout: leetcode-entry
title: "779. K-th Symbol in Grammar"
permalink: "/leetcode/problem/2023-10-25-779-k-th-symbol-in-grammar/"
leetcode_ui: true
entry_slug: "2023-10-25-779-k-th-symbol-in-grammar"
---

[779. K-th Symbol in Grammar](https://leetcode.com/problems/k-th-symbol-in-grammar/description/) medium
[blog post](https://leetcode.com/problems/k-th-symbol-in-grammar/solutions/4205798/kotlin-subproblem/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/25102023-779-k-th-symbol-in-grammar?r=2bam17&utm_campaign=post&utm_medium=web)
![image.png](/assets/leetcode_daily_images/8ff9a12f.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/381

#### Problem TLDR

Binary Tree `0 -> 01`, `1 -> 10` at `[n][k]` position

#### Intuition

Let's draw the example and see the pattern:

```kotlin
  //1                                    [0]
  //2                  [0]                                          1
  //3        [0]                    1                      1                   0
  //4     0       [1]          1         0            1         0          0         1
  //5  0    1    1   [0]     1    0    0    1       1    0    0    1     0    1    1    0
  //6 0 1  1 0  1 0 [0]1    1 0  0 1  0 1  1 0     1 0  0 1  0 1  1 0   0 1  1 0  1 0  0 1
  //  1 2  3 4  5 6  7 8    9
  //                 ^
```

Some observations:

* Every `0` starts its own tree, and every `1` start its own pattern of a tree.
* We can know the position in the previous row: `(k + 1) / 2`
* If previous value is `0`, current pair is `01`, otherwise `10`

#### Approach

* we don't need to memorize the recursion, as it goes straightforward up
* we can use `and 1` bit operation instead of `% 2`

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun kthGrammar(n: Int, k: Int): Int = if (n == 1) 0 else
    (if (kthGrammar(n - 1, (k + 1) / 2) == 0) k.inv() else k) and 1

```

