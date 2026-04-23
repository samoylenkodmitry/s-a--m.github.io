---
layout: leetcode-entry
title: "3005. Count Elements With Maximum Frequency"
permalink: "/leetcode/problem/2024-03-08-3005-count-elements-with-maximum-frequency/"
leetcode_ui: true
entry_slug: "2024-03-08-3005-count-elements-with-maximum-frequency"
---

[3005. Count Elements With Maximum Frequency](https://leetcode.com/problems/count-elements-with-maximum-frequency/description/) easy
[blog post](https://leetcode.com/problems/count-elements-with-maximum-frequency/solutions/4841086/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/08032024-3005-count-elements-with?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/teYJDw4W-pE)
![image.png](/assets/leetcode_daily_images/652ef615.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/532

#### Problem TLDR

Count of max-freq nums #easy

#### Intuition

Count frequencies, then filter by max and sum.

#### Approach

There are at most `100` elements, we can use array to count.

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

  fun maxFrequencyElements(nums: IntArray) = nums
  .asList().groupingBy { it }.eachCount().values.run {
    val max = maxOf { it }
    sumBy { if (it < max) 0 else it }
  }

```
```rust

  pub fn max_frequency_elements(nums: Vec<i32>) -> i32 {
    let mut freq = vec![0i32; 101];
    for x in nums { freq[x as usize] += 1; }
    let max = freq.iter().max().unwrap();
    freq.iter().filter(|&f| f == max).sum()
  }

```

