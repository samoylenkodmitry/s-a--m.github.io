---
layout: leetcode-entry
title: "2038. Remove Colored Pieces if Both Neighbors are the Same Color"
permalink: "/leetcode/problem/2023-10-02-2038-remove-colored-pieces-if-both-neighbors-are-the-same-color/"
leetcode_ui: true
entry_slug: "2023-10-02-2038-remove-colored-pieces-if-both-neighbors-are-the-same-color"
---

[2038. Remove Colored Pieces if Both Neighbors are the Same Color](https://leetcode.com/problems/remove-colored-pieces-if-both-neighbors-are-the-same-color/description/) medium
[blog post](https://leetcode.com/problems/remove-colored-pieces-if-both-neighbors-are-the-same-color/solutions/4117386/kotlin-sliding-window/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/2102023-2038-remove-colored-pieces?r=2bam17&utm_campaign=post&utm_medium=web)
![image.png](/assets/leetcode_daily_images/0b855093.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/357

#### Problem TLDR

Is `A` wins in middle-removing `AAA` or `BBB` game

#### Intuition

We quickly observe, that removing `A` in `BBAAABB` doesn't make `B` turn possible, so the outcome does not depend on how exactly positions are removed. `A` can win if it's possible game turns are more than `B`. So, the problem is to find how many consequent `A`'s and `B`'s are.

#### Approach

We can count `A` and `B` in a single pass, however, let's write a two-pass one-liner using `window` Kotlin method.

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$, can be O(1) if `asSequence` used

#### Code

```kotlin

    fun winnerOfGame(colors: String) = with(colors.windowed(3)) {
      count { it.all { it == 'A' } } > count { it.all { it == 'B' } }
    }

```

