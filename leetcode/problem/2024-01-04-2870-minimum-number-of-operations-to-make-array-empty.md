---
layout: leetcode-entry
title: "2870. Minimum Number of Operations to Make Array Empty"
permalink: "/leetcode/problem/2024-01-04-2870-minimum-number-of-operations-to-make-array-empty/"
leetcode_ui: true
entry_slug: "2024-01-04-2870-minimum-number-of-operations-to-make-array-empty"
---

[2870. Minimum Number of Operations to Make Array Empty](https://leetcode.com/problems/minimum-number-of-operations-to-make-array-empty/description/) medium
[blog post](https://leetcode.com/problems/minimum-number-of-operations-to-make-array-empty/solutions/4504248/kotlin-from-dp-to-math/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/4012024-2870-minimum-number-of-operations?r=2bam17&utm_campaign=post&utm_medium=web&showWelcome=true)
[youtube](https://youtu.be/EdERXCDnyF8)
![image.png](/assets/leetcode_daily_images/cb61adbd.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/461

#### Problem TLDR

Minimum pairs or triples duplicate removal operations to empty array of numbers.

#### Intuition

The first idea, is to count each kind of number. Then we must analyze each `frequency`: the number of removal operations `ops` will be the same for each `f`, so we can write a Dynamic Programming recurrent formula: `ops(f) = 1 + min(ops(f - 2), ops(f - 3))`. This is an accepted solution.

Then, we can think about other ways to optimally split `f` into a sum of `a*2 + b*3`: we must maximize `b` and minimize `a`. To do that, let's prioritize `f % 3 == 0` check. Our checks will be in this order:
```
f % 3 == 0 -> f / 3
(f - 2) % 3 == 0 -> 1 + f / 2
((f - 2) - 2) % 3 == 0 -> 1 + f / 2
... and so on
```
However, we can spot that recurrence repeat itself like this: `f, f - 2, f - 4, f - 6, ...`. As `6` is also divisible by `3`, there are total three checks needed: `f % 3, (f - 2) % 3 and (f - 4) % 3`.

#### Approach

Write the recurrent DFS function, then add a HashMap cache, then optimize everything out.
Use the Kotlin's API:
* groupBy
* mapValues
* sumOf

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

  fun minOperations(nums: IntArray) = nums
    .groupBy { it }.mapValues { it.value.size }.values
    .sumOf { f -> when {
      f < 2 -> return -1
      f % 3 == 0 -> f / 3
      (f - 2) % 3 == 0 || (f - 4) % 3 == 0 -> 1 + f / 3
      else -> return -1
    }}

```

