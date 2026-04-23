---
layout: leetcode-entry
title: "3068. Find the Maximum Sum of Node Values"
permalink: "/leetcode/problem/2024-05-19-3068-find-the-maximum-sum-of-node-values/"
leetcode_ui: true
entry_slug: "2024-05-19-3068-find-the-maximum-sum-of-node-values"
---

[3068. Find the Maximum Sum of Node Values](https://leetcode.com/problems/find-the-maximum-sum-of-node-values/description/) hard
[blog post](https://leetcode.com/problems/find-the-maximum-sum-of-node-values/solutions/5178257/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/19052024-3068-find-the-maximum-sum?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/3vk2zlIMUww)
![2024-05-19_11-13.webp](/assets/leetcode_daily_images/fdccbed0.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/607

#### Problem TLDR

Max sum after `xor k` any edges in a tree #hard #math

#### Intuition

Let's just draw and try to build an intuition.
![2024-05-19_09-10.webp](/assets/leetcode_daily_images/f1b35be2.webp)
![2024-05-19_09-21.webp](/assets/leetcode_daily_images/e51a0baf.webp)
We can cancel out `xor` if we apply an even number of times.

This is where I was stuck and gave up after trying to build the DP solution.

Now, the actual solution: we can cancel out `all` xor between any two nodes: `a-b-c-d, a^k-b^k-c-d, a^k-b-c^k-d, a^k-b-c-d^k`. Effectively, the task now is to do `xor` on all nodes where it gives us increase in the sum.

However, as `xor` must happen in `pairs` we still need to consider how many operations we do. For even just take the sum, but for odd there are `two` cases: flip one xor back, or do one extra xor (that's why we use `abs`). To do the extra flip we must choose the minimum return of the value.

#### Approach

Spend at least 1 hour before giving up.

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun maximumValueSum(nums: IntArray, k: Int, edges: Array<IntArray>): Long {
        var sum = 0L; var xorCount = 0; var minMax = Int.MAX_VALUE / 2
        for (n in nums) {
            sum += max(n, n xor k).toLong()
            if (n xor k > n) xorCount++
            minMax = min(minMax, abs((n xor k) - n))
        }
        return sum - minMax * (xorCount % 2)
    }

```
```rust

    pub fn maximum_value_sum(nums: Vec<i32>, k: i32, edges: Vec<Vec<i32>>) -> i64 {
        let (mut sum, mut cnt, mut min) = (0, 0, i32::MAX);
        for n in nums {
            sum += n.max(n ^ k) as i64;
            if n ^ k > n { cnt += 1 }
            min = min.min(((n ^ k) - n).abs())
        }; sum - (min * (cnt % 2)) as i64
    }

```

