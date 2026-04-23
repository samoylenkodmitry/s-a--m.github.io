---
layout: leetcode-entry
title: "3355. Zero Array Transformation I"
permalink: "/leetcode/problem/2025-05-20-3355-zero-array-transformation-i/"
leetcode_ui: true
entry_slug: "2025-05-20-3355-zero-array-transformation-i"
---

[3355. Zero Array Transformation I](https://leetcode.com/problems/zero-array-transformation-i/description/) medium
[blog post](https://leetcode.com/problems/zero-array-transformation-i/solutions/6761874/kotlin-rust-by-samoylenkodmitry-u0uv/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/20052025-3355-zero-array-transformation?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/7KNbIUk7PwI)
![1.webp](/assets/leetcode_daily_images/4b5b7c93.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/994

#### Problem TLDR

Can intersecting intervals decrease an array #medium #line_sweep

#### Intuition

Line sweep trick: store `starts` and `ends` of the intervals, then do the line sweep by increasing and decreasing prefix sum.

#### Approach

* decreasing should be `after` the end

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

// 22ms
    fun isZeroArray(nums: IntArray, queries: Array<IntArray>): Boolean {
        val d = IntArray(nums.size + 1); var s = 0
        for ((s, e) in queries) { d[s]++; d[e + 1]-- }
        return nums.zip(d).all { (n, d) -> s += d; s >= n }
    }

```
```kotlin

// 3ms
    fun isZeroArray(nums: IntArray, queries: Array<IntArray>): Boolean {
        val d = IntArray(nums.size + 1); var s = 0
        for ((s, e) in queries) { d[s]++; d[e + 1]-- }
        for ((i, x) in nums.withIndex()) {
            s += d[i]
            if (s < x) return false
        }
        return true
    }

```
```rust

// 5ms
    pub fn is_zero_array(nums: Vec<i32>, queries: Vec<Vec<i32>>) -> bool {
        let (mut d, mut s) = (vec![0; nums.len() + 1], 0);
        for q in queries { d[q[0] as usize] += 1; d[q[1] as usize + 1] -= 1 }
        nums.iter().zip(d.iter()).all(|(x, d)| { s += d; s >= *x })
    }

```
```c++

// 0ms https://leetcode.com/problems/zero-array-transformation-i/submissions/1638961889
    bool isZeroArray(vector<int>& nums, vector<vector<int>>& queries) {
        vector<int> d(size(nums) + 1); int s = 0;
        for (auto& q: queries) ++d[q[0]], --d[q[1] + 1];
        for (int i = 0; int x: nums) { s += d[i++]; if (s < x) return 0; }
        return 1;
    }

```

