---
layout: leetcode-entry
title: "1155. Number of Dice Rolls With Target Sum"
permalink: "/leetcode/problem/2023-12-26-1155-number-of-dice-rolls-with-target-sum/"
leetcode_ui: true
entry_slug: "2023-12-26-1155-number-of-dice-rolls-with-target-sum"
---

[1155. Number of Dice Rolls With Target Sum](https://leetcode.com/problems/number-of-dice-rolls-with-target-sum/description/) medium
[blog post](https://leetcode.com/problems/number-of-dice-rolls-with-target-sum/solutions/4459886/kotlin-dp/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/26122023-1155-number-of-dice-rolls?r=2bam17&utm_campaign=post&utm_medium=web&showWelcome=true)
[youtube](https://youtu.be/k2xos_3iM7E)
![image.png](/assets/leetcode_daily_images/89eae29d.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/451

#### Problem TLDR

Ways to throw once `n` dices with `k` faces to make `target` sum.

#### Intuition

Let's consider each dice and try all the possible face. By repeating the process for all the dices, check if the final sum is equal to the target. The result will only depend on the dice count and target sum, so it can be cached.

#### Approach

Write brute force DFS, than add HashMap or array cache.

#### Complexity

- Time complexity:
$$O(nkt)$$, nt - is a DFS search space, k - is the iteration inside

- Space complexity:
$$O(nt)$$

#### Code

```kotlin

  fun numRollsToTarget(n: Int, k: Int, target: Int): Int {
    val dp = mutableMapOf<Pair<Int, Int>, Int>()
    fun dfs(c: Int, s: Int): Int =
      dp.getOrPut(c to s) { when {
          c == 0 -> if (s == 0) 1 else 0
          s <= 0 -> 0
          else -> (1..k).fold(0) { ways, d ->
            (ways + dfs(c - 1, s - d)) % 1_000_000_007
          }
      } }

    return dfs(n, target)
  }

```

