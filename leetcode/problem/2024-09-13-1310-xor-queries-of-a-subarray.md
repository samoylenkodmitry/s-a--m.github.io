---
layout: leetcode-entry
title: "1310. XOR Queries of a Subarray"
permalink: "/leetcode/problem/2024-09-13-1310-xor-queries-of-a-subarray/"
leetcode_ui: true
entry_slug: "2024-09-13-1310-xor-queries-of-a-subarray"
---

[1310. XOR Queries of a Subarray](https://leetcode.com/problems/xor-queries-of-a-subarray/description/) medium
[blog post](https://leetcode.com/problems/xor-queries-of-a-subarray/solutions/5779644/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/13092024-1310-xor-queries-of-a-subarray?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/OCZCdN6uSaU)
![1.webp](/assets/leetcode_daily_images/6a1079cc.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/733

#### Problem TLDR

Run `queries[[from, to]]` of `xor(arr[from..to])` #medium #bit_manipulation

#### Intuition

The `xor` operation is cumulative and associative: swapping and grouping don't matter (like a sum or multiply). So, we can precompute prefix `xor` and use it to compute xor[i..j] in O(1).

#### Approach

* we can reuse the input array

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun xorQueries(arr: IntArray, queries: Array<IntArray>): IntArray {
        for (i in 1..<arr.size) arr[i] = arr[i] xor arr[i - 1]
        return queries.map { (from, to) ->
            arr[to] xor (arr.getOrNull(from - 1) ?: 0)
        }.toIntArray()
    }

```
```rust

    pub fn xor_queries(mut arr: Vec<i32>, queries: Vec<Vec<i32>>) -> Vec<i32> {
        for i in 1..arr.len() { arr[i] ^= arr[i - 1] }
        queries.into_iter().map(|q|
            arr[q[1] as usize] ^ arr[..].get(q[0] as usize - 1).unwrap_or(&0)).collect()
    }

```
```c++

    vector<int> xorQueries(vector<int>& arr, vector<vector<int>>& queries) {
        for (int i = 1; i < arr.size(); i++) arr[i] ^= arr[i - 1];
        vector<int> res; res.reserve(queries.size());
        for (const auto q: queries)
            res.push_back(arr[q[1]] ^ (q[0] > 0 ? arr[q[0] - 1] : 0));
        return res;
    }

```

