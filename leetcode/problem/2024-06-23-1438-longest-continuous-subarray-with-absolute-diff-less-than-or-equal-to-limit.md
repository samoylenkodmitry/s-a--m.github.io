---
layout: leetcode-entry
title: "1438. Longest Continuous Subarray With Absolute Diff Less Than or Equal to Limit"
permalink: "/leetcode/problem/2024-06-23-1438-longest-continuous-subarray-with-absolute-diff-less-than-or-equal-to-limit/"
leetcode_ui: true
entry_slug: "2024-06-23-1438-longest-continuous-subarray-with-absolute-diff-less-than-or-equal-to-limit"
---

[1438. Longest Continuous Subarray With Absolute Diff Less Than or Equal to Limit](https://leetcode.com/problems/longest-continuous-subarray-with-absolute-diff-less-than-or-equal-to-limit/description/) medium
[blog post](https://leetcode.com/problems/longest-continuous-subarray-with-absolute-diff-less-than-or-equal-to-limit/solutions/5355419/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/23062024-1438-longest-continuous?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/E9Z-SU8H0fU)
![2024-06-23_07-19_1.webp](/assets/leetcode_daily_images/0d3037bf.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/648

#### Problem TLDR

Longest subarray with `abs(a[i] - a[j]) <= limit` #medium #sliding_window #monotonic_queue

#### Intuition

Let's observe how we can do this in a single iteration:

```j

    //      0 1 2 3
    //      8 2 4 7    limit=4
    // 0    i
    //      j       8
    // 1      i     8 2    or 2
    // 2        i   8 2 4  8-2=6>4 -> move j
    //        j     2 4
    // 3          i 2 4 7  7-2=5>4 -> move j
    //          j   4 7

```

We should keep the window `j..i` and maintain maximums and minimums.

To find next maximum after current is dropped we can use `Monotonic Queue` technique: make it always decreasing, like `5 4 3 2 1`.
If any new value is bigger then the tail, for example `add 4`, it will be the next maximum and the tail `3 2 1` becomes irrelevant: `5 4 3 2 1 + 4 -> 5 4 4`.

(Another solution would be to just use two heaps, one for maxiums, another for minimums.)

#### Approach

* iterators saves some lines: `maxOf`, `iter().max()`
* notice `unwrap_or(&n)` trick in Rust

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun longestSubarray(nums: IntArray, limit: Int): Int {
        val mins = ArrayDeque<Int>(); val maxs = ArrayDeque<Int>()
        var j = 0
        return nums.withIndex().maxOf { (i, n) ->
            while (mins.size > 0 && mins.last() > n) mins.removeLast()
            while (maxs.size > 0 && maxs.last() < n) maxs.removeLast()
            mins += n; maxs += n
            if (maxs.first() - mins.first() > limit) {
                if (nums[j] == maxs.first()) maxs.removeFirst()
                if (nums[j++] == mins.first()) mins.removeFirst()
            }
            i - j + 1
        }
    }

```
```rust

    pub fn longest_subarray(nums: Vec<i32>, limit: i32) -> i32 {
        let (mut mins, mut maxs, mut j) = (VecDeque::new(), VecDeque::new(), 0);
        nums.iter().enumerate().map(|(i, &n)| {
            while *mins.back().unwrap_or(&n) > n { mins.pop_back(); }
            while *maxs.back().unwrap_or(&n) < n { maxs.pop_back(); }
            mins.push_back(n); maxs.push_back(n);
            if maxs.front().unwrap() - mins.front().unwrap() > limit {
                if nums[j] == *mins.front().unwrap() { mins.pop_front(); }
                if nums[j] == *maxs.front().unwrap() { maxs.pop_front(); }
                j += 1
            }
            (i - j + 1) as i32
        }).max().unwrap()
    }

```

