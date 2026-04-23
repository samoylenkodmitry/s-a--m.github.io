---
layout: leetcode-entry
title: "1647. Minimum Deletions to Make Character Frequencies Unique"
permalink: "/leetcode/problem/2023-09-12-1647-minimum-deletions-to-make-character-frequencies-unique/"
leetcode_ui: true
entry_slug: "2023-09-12-1647-minimum-deletions-to-make-character-frequencies-unique"
---

[1647. Minimum Deletions to Make Character Frequencies Unique](https://leetcode.com/problems/minimum-deletions-to-make-character-frequencies-unique/description/) medium
[blog post](https://leetcode.com/problems/minimum-deletions-to-make-character-frequencies-unique/solutions/4033633/kotlin-collections-api/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/12092023-1647-minimum-deletions-to?r=2bam17&utm_campaign=post&utm_medium=web)
![image.png](/assets/leetcode_daily_images/8a8441c4.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/337

#### Problem TLDR

Minimum removes duplicate frequency chars from string

#### Intuition

```
    // b b c e b a b
    // 1 1 1 4
```

Characters doesn't matter, only frequencies. Let's sort them and scan one-by-one from biggest to small and descriase max value.

#### Approach

Let's use Kotlin collections API:
* groupBy - converts string into groups by characters
* sortedDescending - sorts by descending
* sumBy - iterates over all values and sums the lambda result

#### Complexity
- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun minDeletions(s: String): Int {
      var prev = Int.MAX_VALUE
      return s.groupBy { it }.values
        .map { it.size }
        .sortedDescending()
        .sumBy {
          prev = maxOf(0, minOf(it, prev - 1))
          maxOf(0, it - prev)
        }
    }

```

