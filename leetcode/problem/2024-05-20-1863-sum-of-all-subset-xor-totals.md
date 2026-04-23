---
layout: leetcode-entry
title: "1863. Sum of All Subset XOR Totals"
permalink: "/leetcode/problem/2024-05-20-1863-sum-of-all-subset-xor-totals/"
leetcode_ui: true
entry_slug: "2024-05-20-1863-sum-of-all-subset-xor-totals"
---

[1863. Sum of All Subset XOR Totals](https://leetcode.com/problems/sum-of-all-subset-xor-totals/description/) easy
[blog post](https://leetcode.com/problems/sum-of-all-subset-xor-totals/solutions/5182581/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/20052024-1863-sum-of-all-subset-xor?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/lg23cKE6Jg4)
![2024-05-20_08-11.webp](/assets/leetcode_daily_images/a87425ff.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/608

#### Problem TLDR

Sum of subsets xors #easy #dfs #backtracking

#### Intuition

The problem size is small, only `12` items, we can brute-force the problem. One way is a bitmask from `0` to `2^12`, then each time iterate over array and choose only set bits for indices. This will take O(n2^n) time and O(1) space.
Another way is recursive backtracking: each time make a decision to take item or leave it, adding to the result in the end. This will take O(2^n) time and O(n) space for the recursion depth.

#### Approach

Backtracking code is shorter.
* notice how `slices` are used in Rust

#### Complexity

- Time complexity:
$$O(2^n)$$ `two` decision explorations are made `n` times

- Space complexity:
$$O(n)$$ for the recursion depth

#### Code

```kotlin

    fun subsetXORSum(nums: IntArray): Int {
        fun dfs(i: Int, x: Int): Int = if (i < nums.size)
            dfs(i + 1, x) + dfs(i + 1, x xor nums[i]) else x
        return dfs(0, 0)
    }

```
```rust

    pub fn subset_xor_sum(nums: Vec<i32>) -> i32 {
        fn dfs(n: &[i32], x: i32) -> i32 { if n.len() > 0
            { dfs(&n[1..], x) + dfs(&n[1..], x ^ n[0]) } else { x }
        }
        dfs(&nums, 0)
    }

```

