---
layout: leetcode-entry
title: "2958. Length of Longest Subarray With at Most K Frequency"
permalink: "/leetcode/problem/2024-03-28-2958-length-of-longest-subarray-with-at-most-k-frequency/"
leetcode_ui: true
entry_slug: "2024-03-28-2958-length-of-longest-subarray-with-at-most-k-frequency"
---

[2958. Length of Longest Subarray With at Most K Frequency](https://leetcode.com/problems/length-of-longest-subarray-with-at-most-k-frequency/description) medium
[blog post](https://leetcode.com/problems/length-of-longest-subarray-with-at-most-k-frequency/solutions/4936162/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/28032024-2958-length-of-longest-subarray?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/UzGYOywIPIE)
![2024-03-28_09-04.webp](/assets/leetcode_daily_images/a516cbd7.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/552

#### Problem TLDR

Max subarray length with frequencies <= `k` #medium

#### Intuition

There is a known `sliding window` pattern: right pointer will increase the frequency and left pointer will decrease it. Not try to expand as much as possible, then shrink until conditions are met.

#### Approach

* move the right pointer one position at a time
* we can use `maxOf` in Kotlin or `max` in Rust

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

  fun maxSubarrayLength(nums: IntArray, k: Int): Int {
    val freq = mutableMapOf<Int, Int>(); var j = 0
    return nums.indices.maxOf { i ->
      freq[nums[i]] = 1 + (freq[nums[i]] ?: 0)
      while (freq[nums[i]]!! > k)
        freq[nums[j]] = freq[nums[j++]]!! - 1
      i - j + 1
    }
  }

```
```rust

    pub fn max_subarray_length(nums: Vec<i32>, k: i32) -> i32 {
      let (mut freq, mut j) = (HashMap::new(), 0);
      (0..nums.len()).map(|i| {
        *freq.entry(nums[i]).or_insert(0) += 1;
        while freq[&nums[i]] > k {
          *freq.get_mut(&nums[j]).unwrap() -= 1; j += 1
        }
        i - j + 1
      }).max().unwrap() as i32
    }

```

