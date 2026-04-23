---
layout: leetcode-entry
title: "1630. Arithmetic Subarrays"
permalink: "/leetcode/problem/2023-11-23-1630-arithmetic-subarrays/"
leetcode_ui: true
entry_slug: "2023-11-23-1630-arithmetic-subarrays"
---

[1630. Arithmetic Subarrays](https://leetcode.com/problems/arithmetic-subarrays/description/) medium
[blog post](https://leetcode.com/problems/arithmetic-subarrays/solutions/4319276/kotlin-priorityqueue/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/23112023-1630-arithmetic-subarrays?r=2bam17&utm_campaign=post&utm_medium=web)
![image.png](/assets/leetcode_daily_images/d0d3394f.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/414

#### Problem TLDR

Query array ranges can form arithmetic sequence

#### Intuition

Given the problem contraints, the naive solution would work: just sort the subarray and check the `diff`.

#### Approach

We can use PriorityQueue

#### Complexity

- Time complexity:
$$O(n^2log(n))$$

- Space complexity:
$$O(n)$$

#### Code

```

  fun checkArithmeticSubarrays(nums: IntArray, l: IntArray, r: IntArray) =
  List(l.size) { ind ->
    val pq = PriorityQueue<Int>()
    for (i in l[ind]..r[ind]) pq.add(nums[i])
    val diff = -pq.poll() + pq.peek()
    var prev = pq.poll()
    while (pq.isNotEmpty()) {
      if (pq.peek() - prev != diff) return@List false
      prev = pq.poll()
    }
    true
  }

```

