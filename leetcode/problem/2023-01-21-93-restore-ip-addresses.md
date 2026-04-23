---
layout: leetcode-entry
title: "93. Restore IP Addresses"
permalink: "/leetcode/problem/2023-01-21-93-restore-ip-addresses/"
leetcode_ui: true
entry_slug: "2023-01-21-93-restore-ip-addresses"
---

[93. Restore IP Addresses](https://leetcode.com/problems/restore-ip-addresses/description/) medium

[https://t.me/leetcode_daily_unstoppable/92](https://t.me/leetcode_daily_unstoppable/92)

[blog post](https://leetcode.com/problems/restore-ip-addresses/solutions/3080471/kotlin-dfs-backtracking/)

```kotlin
    fun restoreIpAddresses(s: String): List<String> {
	val res = mutableSetOf<String>()
	fun dfs(pos: Int, nums: MutableList<Int>) {
		if (pos == s.length || nums.size > 4) {
			if (nums.size == 4) res += nums.joinToString(".")
			return
		}
		var n = 0

		for (i in pos..s.lastIndex) {
			n = n*10 + s[i].toInt() - '0'.toInt()
			if (n > 255) break
			nums += n
			dfs(i + 1, nums)
			nums.removeAt(nums.lastIndex)
			if (n == 0) break
		}
	}
	dfs(0, mutableListOf())
	return res.toList()
}

```

So, the size of the problem is small. We can do full DFS.
At every step, choose either take a number or split. Add to the solution if the result is good.

* use set for results
* use backtracking to save some space

Some optimizations:
* exit early when nums.size > 5,
* use math to build a number instead of parsing substring

Space: O(2^N), Time: O(2^N)

