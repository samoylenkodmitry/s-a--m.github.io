---
layout: leetcode-entry
title: "2391. Minimum Amount of Time to Collect Garbage"
permalink: "/leetcode/problem/2023-11-20-2391-minimum-amount-of-time-to-collect-garbage/"
leetcode_ui: true
entry_slug: "2023-11-20-2391-minimum-amount-of-time-to-collect-garbage"
---

[2391. Minimum Amount of Time to Collect Garbage](https://leetcode.com/problems/minimum-amount-of-time-to-collect-garbage/description/) medium
[blog post](https://leetcode.com/problems/minimum-amount-of-time-to-collect-garbage/solutions/4308211/kotlin/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/20112023-2391-minimum-amount-of-time?r=2bam17&utm_campaign=post&utm_medium=web)
![image.png](/assets/leetcode_daily_images/be32f621.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/411

#### Problem TLDR

Time to pick 3-typed `garbage[]` by 3 trucks traveling to the right `travel[]` time

#### Intuition

We can hardcode the algorithm from the description examples, for each truck individually.

#### Approach

Let's try to minify the code:
* all garbage must be picked up, so add `garbage.sumBy { it.length }`
* for each type, truck will travel until the last index with this type

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun garbageCollection(garbage: Array<String>, travel: IntArray): Int =
      garbage.sumBy { it.length } +
        "MPG".sumBy { c ->
          (1..garbage.indexOfLast { c in it }).sumBy { travel[it - 1] }
        }

```

