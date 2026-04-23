---
layout: leetcode-entry
title: "2554. Maximum Number of Integers to Choose From a Range I"
permalink: "/leetcode/problem/2024-12-06-2554-maximum-number-of-integers-to-choose-from-a-range-i/"
leetcode_ui: true
entry_slug: "2024-12-06-2554-maximum-number-of-integers-to-choose-from-a-range-i"
---

[2554. Maximum Number of Integers to Choose From a Range I](https://leetcode.com/problems/maximum-number-of-integers-to-choose-from-a-range-i/description/) medium
[blog post](https://leetcode.com/problems/maximum-number-of-integers-to-choose-from-a-range-i/solutions/6119055/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/06122024-2554-maximum-number-of-integers?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/XOkn48mkCSc)
[deep-dive](https://notebooklm.google.com/notebook/03c5d8b8-ac16-4a40-9d5f-a4f42fca82b8/audio)
![1.webp](/assets/leetcode_daily_images/ac08696a.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/824

#### Problem TLDR

Sum `1..n` excluding `banned` until `maxSum` #medium

#### Intuition

* we can use a HashSet
* we can sort and do two pointers
* we can precompute all sums and do a binary search

#### Approach

* careful with duplicates in the sort solution

#### Complexity

- Time complexity:
$$O(n)$$ or O(nlog(n))

- Space complexity:
$$O(n)$$ or O(1)

#### Code

```kotlin

    fun maxCount(banned: IntArray, n: Int, maxSum: Int): Int {
        val set = banned.toSet(); var s = 0; var cnt = 0
        for (x in 1..n) if (x !in set) {
            s += x; if (s > maxSum) break; cnt++
        }
        return cnt
    }

```
```rust

    pub fn max_count(mut banned: Vec<i32>, n: i32, max_sum: i32) -> i32 {
        banned.sort_unstable(); let (mut j, mut s, mut cnt) = (0, 0, 0);
        for x in 1..=n {
            if j < banned.len() && x == banned[j] {
                while j < banned.len() && x == banned[j] { j += 1 }
              k
        }; cnt
    }

```
```c++

    int maxCount(vector<int>& banned, int n, int maxSum) {
        int cnt = 0, s = 0; int b[10001] = {};
        for (int x: banned) b[x] = 1;
        for (int x = 1; x <= n && s + x <= maxSum; ++x)
            cnt += 1 - b[x], s += x * (1 - b[x]);
        return cnt;
    }

```

