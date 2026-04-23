---
layout: leetcode-entry
title: "2200. Find All K-Distant Indices in an Array"
permalink: "/leetcode/problem/2025-06-24-2200-find-all-k-distant-indices-in-an-array/"
leetcode_ui: true
entry_slug: "2025-06-24-2200-find-all-k-distant-indices-in-an-array"
---

[2200. Find All K-Distant Indices in an Array](https://leetcode.com/problems/find-all-k-distant-indices-in-an-array/description/) easy
[blog post](https://leetcode.com/problems/find-all-k-distant-indices-in-an-array/solutions/6879457/kotlin-rust-by-samoylenkodmitry-fedk/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/24062025-2200-find-all-k-distant?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/W_9D18JkvPc)
![1.webp](/assets/leetcode_daily_images/e0e6f2ed.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1029

#### Problem TLDR

Indices k-distant to key #easy

#### Intuition

The brute force: scan -k..k at each index.
More optimal: build suffix array to predict where is the next key, scan and save the last key position.

#### Approach

* use brute-force for easy problems

#### Complexity

- Time complexity:
$$O(n^2)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

// 32ms
    fun findKDistantIndices(n: IntArray, key: Int, k: Int) =
        n.indices.filter { (max(0, it - k)..min(n.size - 1, it + k)).any { n[it] == key} }

```
```rust

// 1ms
    pub fn find_k_distant_indices(n: Vec<i32>, key: i32, k: i32) -> Vec<i32> {
        (0..n.len() as i32).filter(|i|
        (0.max(i - k)..(i + k + 1).min(n.len() as i32)).any(|j| n[j as usize] == key)).collect()
    }

```

```c++

// 0ms
    vector<int> findKDistantIndices(vector<int>& n, int key, int k) {
        int last = -2 * size(n); vector<int> res{}, next(size(n));
        for (int i = size(n) - 1; i >= 0; --i)
            next[i] = n[i] == key ? i : (i + 1 < size(n) ? next[i + 1] : 2 * size(n));
        for (int i = 0; i < size(n); ++i) {
            if (n[i] == key) last = i;
            if (next[i] - i <= k || i - last <= k) res.push_back(i);
        }
        return res;
    }

```

