---
layout: leetcode-entry
title: "2597. The Number of Beautiful Subsets"
permalink: "/leetcode/problem/2024-05-23-2597-the-number-of-beautiful-subsets/"
leetcode_ui: true
entry_slug: "2024-05-23-2597-the-number-of-beautiful-subsets"
---

[2597. The Number of Beautiful Subsets](https://leetcode.com/problems/the-number-of-beautiful-subsets/description/) medium
[blog post](https://leetcode.com/problems/the-number-of-beautiful-subsets/solutions/5196024/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/23052024-2597-the-number-of-beautiful?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/H_q7_szlo4g)
![2024-05-23_09-05.webp](/assets/leetcode_daily_images/10b229f5.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/611

#### Problem TLDR

Count subsets without `k` difference in them #medium #dfs #backtracking

#### Intuition

There are a DP solutions, but a simple brute-force backtracking is also works. Do a Depth-First search, check element `(n-k)` not added, add element, go deeper, remove element. To get the intuition about how to count subsets, consider this example:
```j
    // 1 1 1 =(111)+(1)+(1)+(1)+(11)+(11)+(11)
```
For each subset of size `n` there are `2^n - 1` subsets. We can sum the on the finish line, or just add on the fly.

One way to optimize this is to use a HashMap and a counter instead of just list.
Another optimization is a bitmask instead of list.

#### Approach

Some tricks here:
* sorting to check just the lower num `n - k`
* `sign` to shorten the `if (size > ) 1 else 0`
* `as i32` do the same in Rust
* `[i32]` slice and `[1..]` next window without the index variable

#### Complexity

- Time complexity:
$$O(n2^n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun beautifulSubsets(nums: IntArray, k: Int): Int {
        val curr = mutableListOf<Int>(); nums.sort()
        fun dfs(i: Int): Int = if (i < nums.size) {
                if ((nums[i] - k) in curr) 0 else {
                    curr += nums[i]; dfs(i + 1).also { curr.removeLast() }
                } + dfs(i + 1)
            } else curr.size.sign
        return dfs(0)
    }

```
```rust

    pub fn beautiful_subsets(mut nums: Vec<i32>, k: i32) -> i32 {
        let mut curr = vec![]; nums.sort_unstable();
        fn dfs(nums: &[i32], curr: &mut Vec<i32>, k: i32) -> i32 {
            if nums.len() > 0 {
                (if curr.contains(&(nums[0] - k)) { 0 } else {
                    curr.push(nums[0]); let r = dfs(&nums[1..], curr, k);
                    curr.pop(); r
                }) + dfs(&nums[1..], curr, k)
            } else { (curr.len() > 0) as i32 }
        } dfs(&nums[..], &mut curr, k)
    }

```

