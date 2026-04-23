---
layout: leetcode-entry
title: "525. Contiguous Array"
permalink: "/leetcode/problem/2024-03-16-525-contiguous-array/"
leetcode_ui: true
entry_slug: "2024-03-16-525-contiguous-array"
---

[525. Contiguous Array](https://leetcode.com/problems/contiguous-array/description/) medium
[blog post](https://leetcode.com/problems/contiguous-array/solutions/4882308/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/16032024-525-contiguous-array?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/ldc2A819Pp8)
![2024-03-16_09-46.jpg](/assets/leetcode_daily_images/0873ecd6.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/540

#### Problem TLDR

Max length of subarray sum(0) == sum(1) #medium

#### Intuition

Let's observe an example `1 0 1 0 0 1 1 0 0 1 0 0 1 0 1`:

```j

  // 0 1 2 3 4 5 6 7 8 91011121314
  // 1 0 1 0 0 1 1 0 0 1 0 0 1 0 1
  // 1 0 1 0-1 0 1 0-1 0-1-2-1-2-1
  // * *0    .           .       .   2
  //   * *1  .           .       .   2
  // * * * *0.           .       .   4
  //         --1         .       .
  // * * * * * *0        .       .   6
  //   * * * * * *1      .       .   6
  // * * * * * * * *0    .       .   8
  //         . * * * *-1 .       .   4
  // * * * * * * * * * *0.       .   10
  //         . * * * * * *-1     .   6
  //         .             --2   .
  //         . * * * * * * * *-1 .   8
  //         .               * *-2   2
  //         . * * * * * * * * * *-1 10 = 14 - 4
  // 0 1 2 3 4 5 6 7 8 91011121314

```

Moving the pointer forward and calculating the `balance` (number of `0` versus number of `1`), we can have compute max length up to the current position in O(1). Just store the first encounter of the `balance` number position.

#### Approach

Let's shorten the code with:
* Kotlin: `maxOf`, `getOrPut`
* Rust: `max`, `entry().or_insert`

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

  fun findMaxLength(nums: IntArray): Int =
    with (mutableMapOf<Int, Int>()) {
      put(0, -1); var b = 0
      nums.indices.maxOf {
        b += if (nums[it] > 0) 1 else -1
        it - getOrPut(b) { it }
      }
    }

```
```rust

  pub fn find_max_length(nums: Vec<i32>) -> i32 {
    let (mut b, mut bToInd) = (0, HashMap::new());
    bToInd.insert(0, -1);
    (0..nums.len() as i32).map(|i| {
      b += if nums[i as usize] > 0 { 1 } else { -1 };
      i - *bToInd.entry(b).or_insert(i)
    }).max().unwrap()
  }

```

