---
layout: leetcode-entry
title: "382. Linked List Random Node"
permalink: "/leetcode/problem/2023-03-10-382-linked-list-random-node/"
leetcode_ui: true
entry_slug: "2023-03-10-382-linked-list-random-node"
---

[382. Linked List Random Node](https://leetcode.com/problems/linked-list-random-node/description/) medium

[blog post](https://leetcode.com/problems/linked-list-random-node/solutions/3279169/kotlin-i-don-t-get-reservior-sampling-just-split-into-buckets-of-size-k/)

```kotlin

class Solution(val head: ListNode?) {
    val rnd = Random(0)
    var curr = head

    fun getRandom(): Int {
        val ind = rnd.nextInt(6)
        var peek: ListNode? = null
        repeat(6) {
            curr = curr?.next
            if (curr == null) curr = head
            if (it == ind) peek = curr
        }

        return peek!!.`val`
    }

}

```

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/144

#### Intuition
Naive solution is trivial. For more interesting solution, you need to look at what others did on leetcode, read an article https://en.wikipedia.org/wiki/Reservoir_sampling and try to understand why it works.

My intuition was: if we need a probability `1/n`, where `n` - is a total number of elements, then what if we split all the input into buckets of size `k`, then choose from every bucket with probability `1/k`. It seems to work, but only for sizes starting from number `6` for the given input.
We just need to be sure, that number of `getRandom` calls are equal to number of buckets `n/k`.

#### Approach
Write the naive solution, then go to Wikipedia, and hope you will not get this in the interview.
#### Complexity
- Time complexity:
$$O(n)$$
- Space complexity:
$$O(1)$$

