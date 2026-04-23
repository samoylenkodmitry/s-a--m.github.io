---
layout: leetcode-entry
title: "886. Possible Bipartition"
permalink: "/leetcode/problem/2022-12-21-886-possible-bipartition/"
leetcode_ui: true
entry_slug: "2022-12-21-886-possible-bipartition"
---

[886. Possible Bipartition](https://leetcode.com/problems/possible-bipartition/description/) medium

[https://t.me/leetcode_daily_unstoppable/59](https://t.me/leetcode_daily_unstoppable/59)

[blog post](https://leetcode.com/problems/possible-bipartition/solutions/2936306/kotlin-union-find/)

```kotlin
fun possibleBipartition(n: Int, dislikes: Array<IntArray>): Boolean {
	val love = IntArray(n+1) { it }
	fun leader(x: Int): Int {
		var i = x
		while (love[i] != i) i = love[i]
		love[x] = i
		return i
	}
	val hate = IntArray(n+1) { -1 }
	dislikes.forEach { (one, two) ->
		val leaderOne = leader(one)
		val leaderTwo = leader(two)
		val enemyOfOne = hate[leaderOne]
		val enemyOfTwo = hate[leaderTwo]
		if (enemyOfOne != -1 && enemyOfOne == enemyOfTwo) return false
		if (enemyOfOne != -1) {
			love[leader(enemyOfOne)] = leaderTwo
		}
		if (enemyOfTwo != -1) {
			love[leader(enemyOfTwo)] = leaderOne
		}
		hate[leaderOne] = leaderTwo
		hate[leaderTwo] = leaderOne
	}
	return true
}

```

We need somehow to union people that hate the same people. We can do it making someone a leader of a group and make just leaders to hate each other.

Keep track of the leaders hating each other in the `hate` array, and people loving their leader in `love` array. (`love` array is basically a Union-Find).
* also use path compression for `leader` method

Space: O(N), Time: O(N) - adding to Union-Find is O(1) amortised

