---
layout: leetcode-entry
title: "368. Largest Divisible Subset"
permalink: "/leetcode/problem/2024-02-09-368-largest-divisible-subset/"
leetcode_ui: true
entry_slug: "2024-02-09-368-largest-divisible-subset"
---

[368. Largest Divisible Subset](https://leetcode.com/problems/largest-divisible-subset/description/) medium
[blog post](https://leetcode.com/problems/largest-divisible-subset/solutions/4700794/kotlin-rust-it-s-a-hard-problem/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/09022024-368-largest-divisible-subset?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/YlXDUoA-XnU)
![image.png](/assets/leetcode_daily_images/3c4f6322.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/500

#### Problem TLDR

Longest subset of divisible by s[i] % s[j] == 0 || s[j] % s[i] == 0.

#### Intuition

Sort always helps, so do it.
Let's imagine a sequence of numbers like this:

```bash
    // 1 3 9 15 27 30 60
    // 1 3 9    27
    // 1 3   15    30 60
    // 3 4 8 16
```
There is a choice to be made: take `9` or `15`. So we can search with DFS and try to take each number.
Also, there are some interesting things happening: for every position there is only one longest suffix subsequence. We can cache it.

#### Approach

I didn't solve it the second time, so I can't give you the working approach yet. Try as hard as you can for 1 hour, then give up and look for solutions.
My problem was: didn't considered DP, but wrote working backtracking solution. Also, I have attempted the graph solution to find a longest path, but that was TLE.

#### Complexity

- Time complexity:
$$O(n^2)$$

- Space complexity:
$$O(n^2)$$

#### Code

```kotlin

  fun largestDivisibleSubset(nums: IntArray): List<Int> {
    nums.sort()
    val dp = mutableMapOf<Int, List<Int>>()
    fun dfs(i: Int): List<Int> = dp.getOrPut(i) {
      var seq = listOf<Int>()
      val x = if (i == 0) 1 else nums[i - 1]
      for (j in i..<nums.size) if (nums[j] % x == 0) {
        val next = listOf(nums[j]) + dfs(j + 1)
        if (next.size > seq.size) seq = next
      }
      seq
    }
    return dfs(0)
  }

```
```rust

    pub fn largest_divisible_subset(mut nums: Vec<i32>) -> Vec<i32> {
      nums.sort_unstable();
      let mut dp: HashMap<usize, Vec<i32>> = HashMap::new();

      fn dfs(nums: &[i32], i: usize, dp: &mut HashMap<usize, Vec<i32>>) -> Vec<i32> {
          dp.get(&i).cloned().unwrap_or_else(|| {
              let x = nums.get(i.wrapping_sub(1)).copied().unwrap_or(1);
              let largest_seq = (i..nums.len())
                  .filter(|&j| nums[j] % x == 0)
                  .map(|j| {
                      let mut next = vec![nums[j]];
                      next.extend(dfs(nums, j + 1, dp));
                      next
                  })
                  .max_by_key(|seq| seq.len())
                  .unwrap_or_else(Vec::new);

              dp.insert(i, largest_seq.clone());
              largest_seq
          })
      }

      dfs(&nums, 0, &mut dp)
    }

```

