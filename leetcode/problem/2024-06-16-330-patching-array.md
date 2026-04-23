---
layout: leetcode-entry
title: "330. Patching Array"
permalink: "/leetcode/problem/2024-06-16-330-patching-array/"
leetcode_ui: true
entry_slug: "2024-06-16-330-patching-array"
---

[330. Patching Array](https://leetcode.com/problems/patching-array/description/) hard
[blog post](https://leetcode.com/problems/patching-array/solutions/5319943/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/16062024-330-patching-array?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/6j95rse_WRI)
![2024-06-16_06-59_1.webp](/assets/leetcode_daily_images/965c04a7.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/641

#### Problem TLDR

Insertions to make subsets sums fill `1..n` #hard

#### Intuition

The hard part for me was to understand range filling law: if range `[1..x]` is filled, then to fill range `[1..x+x]` we can add just one number `x`: it will add all the range of numbers: `1+x, 2+x, 3+x ... x+x`

With this in mind, let's explore example of how to fill the range:

```j
    // 1 5 10      n=20
    // sums = 1, 5, 10, 1+5,1+10,5+10,1+5+10
    // 1 2 3 9
    // 1        [1..1]
    //   2      [..1+2] = [..3]
    //     3    [..3+3] = [..6]
    //       9  9>6+1, 7..9 -> 7 -> [..6+7]= [..13]
    //          [..13+9] = [..22]
    // 1 2 10 20    n=46
    // 1        ..1
    //   2      ..3
    //     10   10>4, ..3+4=..7
    //          10>8, ..7+8=..15
    //          ..15+10=..25
```
When we reach the number `9`, we see the gap between the rightmost border `6` and `9`, so we fill it with the next number after border `7`. After this operation, the filled range becomes `[1..6+7]` and we can take the number `9`.

#### Approach

Look for the tips in the discussion section.

#### Complexity

- Time complexity:
$$O(mlog(n))$$, each time we doubling the border, so it takes `log(n)`

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun minPatches(nums: IntArray, n: Int): Int {
        var count = 0; var border = 0L; var i = 0
        while (border < n) {
            if (i < nums.size && nums[i] <= border + 1) {
                border += nums[i]
                i++
            } else {
                border += border + 1
                count++
            }
        }
        return count
    }

```
```rust

    pub fn min_patches(nums: Vec<i32>, n: i32) -> i32 {
        let (mut border, mut i, mut cnt) = (0, 0, 0);
        while border < n as _ {
            if i < nums.len() && nums[i] as u64 <= border + 1 {
                border += nums[i] as u64;
                i += 1
            } else {
                border += border + 1;
                cnt += 1
            }
        }; cnt
    }

```

