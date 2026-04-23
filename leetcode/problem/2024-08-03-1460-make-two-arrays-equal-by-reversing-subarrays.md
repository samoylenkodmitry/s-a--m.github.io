---
layout: leetcode-entry
title: "1460. Make Two Arrays Equal by Reversing Subarrays"
permalink: "/leetcode/problem/2024-08-03-1460-make-two-arrays-equal-by-reversing-subarrays/"
leetcode_ui: true
entry_slug: "2024-08-03-1460-make-two-arrays-equal-by-reversing-subarrays"
---

[1460. Make Two Arrays Equal by Reversing Subarrays](https://leetcode.com/problems/make-two-arrays-equal-by-reversing-subarrays/description/) easy
[blog post](https://leetcode.com/problems/make-two-arrays-equal-by-reversing-subarrays/solutions/5577994/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/03082024-1460-make-two-arrays-equal?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/sEkfoM-0bAE)
![2024-08-03_09-07_1.webp](/assets/leetcode_daily_images/829e7eac.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/691

#### Problem TLDR

Can `arr` transform to `target` by rotating subarrays? #easy

#### Intuition

By swapping every subarray we can move any position to any other position, effectively sorting the array as we want. So, compare the sorted arrays or compare the numbers' frequencies.

#### Approach

Let's implement both variants.

#### Complexity

- Time complexity:
$$O(n)$$ and $$O(nlogn)$$ for sorting

- Space complexity:
$$O(n)$$ and $$O(1)$$ for sorting

#### Code

```kotlin

    fun canBeEqual(target: IntArray, arr: IntArray) =
        target.groupBy { it } == arr.groupBy { it }

```
```rust

    pub fn can_be_equal(mut target: Vec<i32>, mut arr: Vec<i32>) -> bool {
        target.sort_unstable(); arr.sort_unstable(); target == arr
    }

```

