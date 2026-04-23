---
layout: leetcode-entry
title: "930. Binary Subarrays With Sum"
permalink: "/leetcode/problem/2024-03-14-930-binary-subarrays-with-sum/"
leetcode_ui: true
entry_slug: "2024-03-14-930-binary-subarrays-with-sum"
---

[930. Binary Subarrays With Sum](https://leetcode.com/problems/binary-subarrays-with-sum/description/) medium
[blog post](https://leetcode.com/problems/binary-subarrays-with-sum/solutions/4873512/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/14032024-930-binary-subarrays-with?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/C-y7qYgqqxM)
![2024-03-14_09-06.jpg](/assets/leetcode_daily_images/8879d999.webp)
https://youtu.be/C-y7qYgqqxM
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/538

#### Problem TLDR

Count `goal`-sum subarrays in a `0-1` array #medium

#### Intuition

Let's observe an example:

```j
    // [0,0,1,0,1,0,0,0]
    //1     * * *
    //2   *
    //3 * *
    //4           *
    //5           * *
    //6           * * *
    //7   *       *
    //8 * *       *
    //9   *       * *
    //10* *       * *
    //11  *       * * *
    //12* *       * * *
    // 1 + 2 + 3 + 2*3
```
As we count possible subarrays, we see that zeros suffix and prefix matters and we can derive the math formula for them.
The corner case is an all-zero array: we just take an arithmetic progression sum.

#### Approach

* careful with pointers, widen zeros in a separate step
* use a separate variables to count zeros
* move pointers only forward
* check yourself on the corner cases `0, 0` and `0, 0, 1`

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

  fun numSubarraysWithSum(nums: IntArray, goal: Int): Int {
    var i = 0; var j = 0; var sum = 0; var res = 0
    while (i < nums.size) {
      sum += nums[i]
      while (sum > goal && j < i) sum -= nums[j++]
      if (sum == goal) {
        var z1 = 0
        while (i + 1 < nums.size && nums[i + 1] == 0) { i++; z1++ }
        res += if (goal == 0) (z1 + 1) * (z1 + 2) / 2 else {
          var z2 = 0
          while (j < i && nums[j] == 0) { j++; z2++ }
          1 + z1 + z2 + z1 * z2
        }
      }; i++
    }; return res
  }

```
```rust

  pub fn num_subarrays_with_sum(nums: Vec<i32>, goal: i32) -> i32 {
    let (mut i, mut j, mut sum, mut res) = (0, 0, 0, 0);
    while i < nums.len() {
      sum += nums[i];
      while sum > goal && j < i { sum -= nums[j]; j += 1 }
      if sum == goal {
        let mut z1 = 0;
        while i + 1 < nums.len() && nums[i + 1] == 0 { i += 1; z1 += 1 }
        res += if goal == 0 { (z1 + 1) * (z1 + 2) / 2 } else {
          let mut z2 = 0;
          while j < i && nums[j] == 0 { j += 1; z2 += 1 }
          1 + z1 + z2 + z1 * z2
        }
      }; i += 1
    }; res
  }

```

