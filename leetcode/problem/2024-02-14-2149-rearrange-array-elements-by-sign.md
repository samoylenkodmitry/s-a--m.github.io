---
layout: leetcode-entry
title: "2149. Rearrange Array Elements by Sign"
permalink: "/leetcode/problem/2024-02-14-2149-rearrange-array-elements-by-sign/"
leetcode_ui: true
entry_slug: "2024-02-14-2149-rearrange-array-elements-by-sign"
---

[2149. Rearrange Array Elements by Sign](https://leetcode.com/problems/rearrange-array-elements-by-sign/description/) medium
[blog post](https://leetcode.com/problems/rearrange-array-elements-by-sign/solutions/4724868/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/14022024-2149-rearrange-array-elements?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/Wv3Rw7Jit34)
![image.png](/assets/leetcode_daily_images/3dbdc826.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/505

#### Problem TLDR

Rearrange array to positive-negative sequence.

#### Intuition

First is to understand that we can't do this in-place: for example `1 1 1 1 1 1 -1 -1 -1 -1 -1 -1` we must store somewhere the `1`s that is changed by `-1`s.
Next, just use two pointers and a separate result array.

#### Approach

We can use ping-pong technique for pointers and make work with only the current pointer.
Some language's APIs:
* Kotlin: `indexOfFirst`, `also`, `find`
* Rust: `iter`, `position`, `find`

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

  fun rearrangeArray(nums: IntArray): IntArray {
    var i = nums.indexOfFirst { it > 0 }
    var j = nums.indexOfFirst { it < 0 }
    return IntArray(nums.size) {
      nums[i].also { n ->
        i = (i + 1..<nums.size)
          .find { n > 0 == nums[it] > 0 } ?: 0
        i = j.also { j = i }
      }
    }
  }

```
```rust

  pub fn rearrange_array(nums: Vec<i32>) -> Vec<i32> {
    let mut i = nums.iter().position(|&n| n > 0).unwrap();
    let mut j = nums.iter().position(|&n| n < 0).unwrap();
    (0..nums.len()).map(|_| {
      let n = nums[i];
      i = (i + 1..nums.len())
        .find(|&i| (n > 0) == (nums[i] > 0)).unwrap_or(0);
      (i, j) = (j, i); n
    }).collect()
  }

```

