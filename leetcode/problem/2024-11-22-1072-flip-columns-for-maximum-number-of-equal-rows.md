---
layout: leetcode-entry
title: "1072. Flip Columns For Maximum Number of Equal Rows"
permalink: "/leetcode/problem/2024-11-22-1072-flip-columns-for-maximum-number-of-equal-rows/"
leetcode_ui: true
entry_slug: "2024-11-22-1072-flip-columns-for-maximum-number-of-equal-rows"
---

[1072. Flip Columns For Maximum Number of Equal Rows](https://leetcode.com/problems/flip-columns-for-maximum-number-of-equal-rows/description/) medium
[blog post](https://leetcode.com/problems/flip-columns-for-maximum-number-of-equal-rows/solutions/6071515/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/22112024-1072-flip-columns-for-maximum?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/UA_eFpj1mHk)
[deep-dive](https://notebooklm.google.com/notebook/77eccce0-0d43-418d-8308-16a6cbfa8bac/audio)
![1.webp](/assets/leetcode_daily_images/9009ec6a.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/808

#### Problem TLDR

Max same-bit-rows after flipping columns in 01-2D matrix #medium #matrix

#### Intuition

Let's observe what's happening:

```j

    // 0 0 0   ---
    // 0 0 1   ff- or --f   <-- mask
    // 1 1 0   ff- or --f
    // 1 1 1   ---
    //   v
    // 0 0 1
    // 0 0 0
    // 1 1 1
    // 1 1 0
    //     *
    //
    // 0 1 0
    // 0 1 1
    // 1 0 0
    // 1 0 1
    //   *    <-- intermediate column flips are irrelevant

    // 0 0 0 0 0 *
    // 1 1 1 1 1 *
    // 0 0 0 0 1  *
    // 1 1 1 1 0  *
    // 0 0 0 1 1   *     <-- symmetry
    // 1 1 1 0 0   *     <-- symmetry
    // 0 0 1 1 1    *
    // 1 1 0 0 0    *
    // 0 1 1 1 1     *
    // 1 0 0 0 0     *

```

Some observations:
* intermediate flips are irrelevant, only pattern-flips can improve the situation
* each row has a pattern and this pattern has a symmetry with its inverted version
* the pattern and its inversion forms a groups, group size is the answer

#### Approach

* one trick to collapse pattern with its inversion is to xor each with the first bit (@lee's brain idea)
* in Kotlin, simple groupBy works faster than strings hashes
* in Rust [u8] key is faster than `[u128, u128, u128]`, or `[u64; 5]` keys
* c++ has bitset built-in

#### Complexity

- Time complexity:
$$O(nm)$$

- Space complexity:
$$O(nm)$$

#### Code

```kotlin

    fun maxEqualRowsAfterFlips(matrix: Array<IntArray>) =
        matrix.groupBy { r -> r.map { it xor r[0] }}
        .maxOf { it.value.size }

```
```rust

    pub fn max_equal_rows_after_flips(matrix: Vec<Vec<i32>>) -> i32 {
        *matrix.iter().fold(HashMap::new(), |mut hm, r| {
            *hm.entry(r.iter().map(|&b| r[0] ^ b).collect::<Vec<_>>())
            .or_insert(0) += 1; hm
        }).values().max().unwrap() as i32
    }

```
```c++

    int maxEqualRowsAfterFlips(vector<vector<int>>& m) {
        unordered_map<bitset<300>, int>c; int r = 0;
        for (auto v: m) {
            bitset<300>b;
            for (int i = 0; i < size(v);) b[i] = v[0] ^ v[i++];
            r = max(r, ++c[b]);
        }
        return r;
    }

```

