---
layout: leetcode-entry
title: "2044. Count Number of Maximum Bitwise-OR Subsets"
permalink: "/leetcode/problem/2024-10-18-2044-count-number-of-maximum-bitwise-or-subsets/"
leetcode_ui: true
entry_slug: "2024-10-18-2044-count-number-of-maximum-bitwise-or-subsets"
---

[2044. Count Number of Maximum Bitwise-OR Subsets](https://leetcode.com/problems/count-number-of-maximum-bitwise-or-subsets/description/) medium
[blog post](https://leetcode.com/problems/count-number-of-maximum-bitwise-or-subsets/solutions/5930477/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/18102024-2044-count-number-of-maximum?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/5s8UMzStzZk)
[deep-dive](https://notebooklm.google.com/notebook/af67e6ef-04a2-477a-aecb-c38c6bbd7134/audio)
![1.webp](/assets/leetcode_daily_images/f229c56a.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/772

#### Problem TLDR

Count subsequences with max bitwise `or` #medium #backtracking

#### Intuition

The problem size is only `16` elements, so we can do a full Depth-First Search.
First, precompute the target `or`-operation result: it can only increase with each new num added. (we are adding new bits, but never remove)
Then, for each position we can `take` element or `skip` it. The final condition will be `0` or `1` if mask is equal to target.

#### Approach

* we can do a `for` loop inside a DFS, doing skipping positions naturally, have to consider the intermediate target however

#### Complexity

- Time complexity:
$$O(2^n)$$

- Space complexity:
$$O(n)$$ for the recursion depth

#### Code

```kotlin

    fun countMaxOrSubsets(nums: IntArray): Int {
        val maxor = nums.fold(0) { r, t -> r or t }
        fun dfs(i: Int, mask: Int): Int = (if (mask == maxor) 1 else 0) +
            (i..<nums.size).sumOf { j -> dfs(j + 1, mask or nums[j]) }
        return dfs(0, 0)
    }

```
```rust

    pub fn count_max_or_subsets(nums: Vec<i32>) -> i32 {
        let mut or = nums.iter().fold(0, |r, &t| r | t);
        fn dfs(nums: &[i32], m: i32, or: i32) -> i32 {
            if nums.len() == 0 { (m == or) as i32 }
            else { dfs(&nums[1..], m | nums[0], or) + dfs(&nums[1..], m, or) }
        }
        dfs(&nums[..], 0, or)
    }

```
```c++

    int countMaxOrSubsets(vector<int>& nums) {
        int maxor = accumulate(nums.begin(), nums.end(), 0, bit_or<>());
        function<int(int, int)>dfs = [&](int i, int mask) {
            return i == nums.size() ? mask == maxor :
                dfs(i + 1, mask | nums[i]) + dfs(i + 1, mask);
        };
        return dfs(0, 0);
    }

```

