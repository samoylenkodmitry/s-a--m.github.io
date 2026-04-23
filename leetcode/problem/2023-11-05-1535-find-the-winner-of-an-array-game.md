---
layout: leetcode-entry
title: "1535. Find the Winner of an Array Game"
permalink: "/leetcode/problem/2023-11-05-1535-find-the-winner-of-an-array-game/"
leetcode_ui: true
entry_slug: "2023-11-05-1535-find-the-winner-of-an-array-game"
---

[1535. Find the Winner of an Array Game](https://leetcode.com/problems/find-the-winner-of-an-array-game/description/) medium
[blog post](https://leetcode.com/problems/find-the-winner-of-an-array-game/solutions/4250991/kotlin/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/05112023-1535-find-the-winner-of?r=2bam17&utm_campaign=post&utm_medium=web)
![image.png](/assets/leetcode_daily_images/82ed5e8d.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/394

#### Problem TLDR

Find maximum of the `k` nearest in array

#### Intuition

Looking at some examples:

```kotlin
  // 0 1 2 3 4 5
  // 1 3 2 5 4 10            3
  //   3 2 5 4 10 1          3
  //   3   5 4 10 1 2        5
  //       5 4 10 1 2 3      5
  //       5   10 1 2 3 4    10
  //           10 1 2 3 4 5  10 ...
```
we can deduce that the problem is trivial when `k >= arr.size` - it is just a maximum.
Now, when `k < arr.size` we can just simulate the given algorithm and stop on the first `k`-winner.

#### Approach

* we can iterate over `1..arr.lastIndex` or use a clever initialization `wins = -1`

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

  fun getWinner(arr: IntArray, k: Int): Int {
    var wins = -1
    var max = arr[0]
    for (x in arr) {
      if (x > max) {
        wins = 1
        max = x
      } else wins++
      if (wins == k) break
    }
    return max
  }

```

