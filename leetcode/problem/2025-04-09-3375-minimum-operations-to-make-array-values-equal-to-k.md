---
layout: leetcode-entry
title: "3375. Minimum Operations to Make Array Values Equal to K"
permalink: "/leetcode/problem/2025-04-09-3375-minimum-operations-to-make-array-values-equal-to-k/"
leetcode_ui: true
entry_slug: "2025-04-09-3375-minimum-operations-to-make-array-values-equal-to-k"
---

[3375. Minimum Operations to Make Array Values Equal to K](https://leetcode.com/problems/minimum-operations-to-make-array-values-equal-to-k/description/) easy
[blog post](https://leetcode.com/problems/minimum-operations-to-make-array-values-equal-to-k/solutions/6631941/kotlin-rust-by-samoylenkodmitry-ytzz/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/09042025-3375-minimum-operations?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/zr1UHp2d9R8)
![1.webp](/assets/leetcode_daily_images/545bc7ae.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/953

#### Problem TLDR

Count number bigger than k #easy

#### Intuition

The problem description is the hardest part.
The brute-force simulation would be like this: remove largest, skip duplicates, repeat.

#### Approach

* we can use a counting array or a bitset

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun minOperations(n: IntArray, k: Int) =
    if (n.min() < k) -1 else n.toSet().count { it > k }

```
```rust

    pub fn min_operations(n: Vec<i32>, k: i32) -> i32 {
        let mut s = [0; 101];
        for x in n {
            if x < k { return -1 }
            if x > k { s[x as usize] = 1 }
        } s.iter().sum()
    }

```
```c++

    int minOperations(vector<int>& n, int k) {
        bitset<101> b;
        for (int x: n) if (x < k) return -1; else b[x] = b[x] | x > k;
        return b.count();
    }

```

