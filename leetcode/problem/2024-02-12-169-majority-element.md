---
layout: leetcode-entry
title: "169. Majority Element"
permalink: "/leetcode/problem/2024-02-12-169-majority-element/"
leetcode_ui: true
entry_slug: "2024-02-12-169-majority-element"
---

[169. Majority Element](https://leetcode.com/problems/majority-element/description/) easy
[blog post](https://leetcode.com/problems/majority-element/solutions/4714171/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/12022024-169-majority-element?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/EBRvIXGUgKA)
![image.png](/assets/leetcode_daily_images/a5028363.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/503

#### Problem TLDR

Element with frequency > nums.len / 2.

#### Intuition

First thing is to understand the problem, as we need to find not only the most frequent element, but frequency is given > nums.len / 2 by the input constraints.
Next, let's observe examples:
![image.png](/assets/leetcode_daily_images/e2b98606.webp)
There are properties derived from the observation:
* sequence can spread other elements between the common
* common can exist in several islands
* the second common island size is less than first common
* island can be single one
We can write an ugly algorithm full of 'ifs' now.

```kotlin

  fun majorityElement(nums: IntArray): Int {
    var a = -1
    var b = -1
    var countA = 1
    var countB = 0
    var currCount = 1
    var prev = -1
    for (x in nums) {
      if (x == prev) {
        currCount++
        if (currCount > nums.size / 2) return x
      } else {
        if (currCount > 1) {
          if (a == -1) a = prev
          else if (b == -1) b = prev
          if (prev == a) {
            countA += currCount
          }
          if (prev == b) {
            countB += currCount
          }
        }
        currCount = 1
      }
      prev = x
    }
    if (a == -1) a = prev
    else if (b == -1) b = prev
    if (prev == a) {
      countA += currCount
    } else if (prev == b) {
      countB += currCount
    }
    return if (a == -1 && b == -1) {
      nums[0]
    } else if (countA > countB) a else b
  }

```

#### Approach

However, for our pleasure, there is a comment section of leetcode exists, find some big head solution there: it works like a magic for me still. Count the current frequency and decrease it by all others. If others are sum up to a bigger value, our candidate is not the hero.

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

  fun majorityElement(nums: IntArray): Int {
    var a = -1
    var c = 0
    for (x in nums) {
      if (c == 0) a = x
      c += if (x == a) 1 else -1
    }
    return a
  }

```
```rust

  pub fn majority_element(nums: Vec<i32>) -> i32 {
    let (mut a, mut c) = (-1, 0);
    for x in nums {
      if c == 0 { a = x }
      c += if x == a { 1 } else { -1 }
    }
    a
  }

```

